struct BatchMatrixAddMeta {
    M: u32,
    N: u32,
    K: u32,
    batch_count: u32,
    stride_A_batch: u32,
    stride_B_batch: u32,
    stride_C_batch: u32,
    stride_bias_batch: u32,
    bias_mode: u32,     // 0 = None, 1 = 1D (N,), 2 = 2D (M, N), 3 = 3D (B, M, N)
    transpose_A: u32,   // 0 = False, 1 = True
    transpose_B: u32,   // 0 = False, 1 = True
    _pad0: u32,         // Padding to 48 bytes (12 x u32)
};

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<uniform> metamat: BatchMatrixAddMeta;

const TILE_DIM: u32 = 64u;
const TILE_K: u32 = 16u;

// Padded shared memory arrays (+1 float to eliminate bank conflicts)
var<workgroup> tileA: array<array<f32, 17>, 64>;
var<workgroup> tileB: array<array<f32, 65>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let batch_idx = workgroup_id.z;
    if (batch_idx >= metamat.batch_count) {
        return;
    }

    let base_A = batch_idx * metamat.stride_A_batch;
    let base_B = batch_idx * metamat.stride_B_batch;
    let base_C = batch_idx * metamat.stride_C_batch;

    let tiles_x = (metamat.N + TILE_DIM - 1u) / TILE_DIM;
    let tiles_y = (metamat.M + TILE_DIM - 1u) / TILE_DIM;
    let total_tiles = tiles_x * tiles_y;

    // Grid-stride loop over 64x64 output tiles
    var tile_idx = workgroup_id.y * num_workgroups.x + workgroup_id.x;
    let grid_stride = num_workgroups.x * num_workgroups.y;

    while (tile_idx < total_tiles) {
        let tile_y = tile_idx / tiles_x;
        let tile_x = tile_idx % tiles_x;

        let row_start = tile_y * TILE_DIM + local_id.y * 4u;
        let col_start = tile_x * TILE_DIM + local_id.x * 4u;

        var acc0: vec4<f32> = vec4<f32>(0.0);
        var acc1: vec4<f32> = vec4<f32>(0.0);
        var acc2: vec4<f32> = vec4<f32>(0.0);
        var acc3: vec4<f32> = vec4<f32>(0.0);

        let num_k_tiles = (metamat.K + TILE_K - 1u) / TILE_K;

        for (var t = 0u; t < num_k_tiles; t = t + 1u) {

            // 1. Load Tile A
            for (var i = 0u; i < 4u; i = i + 1u) {
                let a_row = tile_y * TILE_DIM + local_id.y * 4u + i;
                let a_col = t * TILE_K + local_id.x;
                if (a_row < metamat.M && a_col < metamat.K) {
                    let a_idx = select(
                        base_A + a_row * metamat.K + a_col,
                        base_A + a_col * metamat.M + a_row,
                        metamat.transpose_A == 1u
                    );
                    tileA[local_id.y * 4u + i][local_id.x] = A[a_idx];
                } else {
                    tileA[local_id.y * 4u + i][local_id.x] = 0.0;
                }
            }

            // 2. Load Tile B
            for (var j = 0u; j < 4u; j = j + 1u) {
                let b_row = t * TILE_K + local_id.y;
                let b_col = tile_x * TILE_DIM + local_id.x * 4u + j;
                if (b_row < metamat.K && b_col < metamat.N) {
                    let b_idx = select(
                        base_B + b_row * metamat.N + b_col,
                        base_B + b_col * metamat.K + b_row,
                        metamat.transpose_B == 1u
                    );
                    tileB[local_id.y][local_id.x * 4u + j] = B[b_idx];
                } else {
                    tileB[local_id.y][local_id.x * 4u + j] = 0.0;
                }
            }

            workgroupBarrier();

            // 3. Compute 4x4 Outer-Product Accumulation
            for (var k = 0u; k < TILE_K; k = k + 1u) {
                let b_vec = vec4<f32>(
                    tileB[k][local_id.x * 4u + 0u],
                    tileB[k][local_id.x * 4u + 1u],
                    tileB[k][local_id.x * 4u + 2u],
                    tileB[k][local_id.x * 4u + 3u]
                );

                let a0 = tileA[local_id.y * 4u + 0u][k];
                let a1 = tileA[local_id.y * 4u + 1u][k];
                let a2 = tileA[local_id.y * 4u + 2u][k];
                let a3 = tileA[local_id.y * 4u + 3u][k];

                acc0 = fma(vec4<f32>(a0), b_vec, acc0);
                acc1 = fma(vec4<f32>(a1), b_vec, acc1);
                acc2 = fma(vec4<f32>(a2), b_vec, acc2);
                acc3 = fma(vec4<f32>(a3), b_vec, acc3);
            }

            workgroupBarrier();
        }

        // 4. Write 4x4 Sub-Block to Output + Fused Bias
        let rows = vec4<u32>(row_start, row_start + 1u, row_start + 2u, row_start + 3u);
        let accs = array<vec4<f32>, 4>(acc0, acc1, acc2, acc3);

        for (var r = 0u; r < 4u; r = r + 1u) {
            let global_r = rows[r];
            if (global_r < metamat.M) {
                for (var c = 0u; c < 4u; c = c + 1u) {
                    let global_c = col_start + c;
                    if (global_c < metamat.N) {
                        var bias_val: f32 = 0.0;
                        if (metamat.bias_mode == 1u) {
                            bias_val = bias[global_c];
                        } else if (metamat.bias_mode == 2u) {
                            bias_val = bias[global_r * metamat.N + global_c];
                        } else if (metamat.bias_mode == 3u) {
                            bias_val = bias[batch_idx * metamat.stride_bias_batch + global_r * metamat.N + global_c];
                        }

                        C[base_C + global_r * metamat.N + global_c] = accs[r][c] + bias_val;
                    }
                }
            }
        }

        tile_idx += grid_stride;
    }
}