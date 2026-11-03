struct BroadcastMeta {
    numel: u32,             // Total elements in target tensor A
    rank: u32,              // Number of active dimensions (1 to 6)
    op_type: u32,           // 0 = Add, 1 = Sub, 2 = Mul, 3 = Div
    _pad0: u32,
    out_shape: array<u32, 6>,
    strides_A: array<u32, 6>,
    strides_B: array<u32, 6>,
};

@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read> metamatrix: BroadcastMeta;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total_elements = metamatrix.numel;
    let stride = num_workgroups.x * WORKGROUP_SIZE;

    var idx = global_id.x;
    while (idx < total_elements) {
        if (idx >= arrayLength(&A)) { break; }
        var temp_idx = idx;
        var offset_A = 0u;
        var offset_B = 0u;

        // Decompose linear index into coordinates from right to left
        for (var i = 0u; i < metamatrix.rank; i = i + 1u) {
            let dim_idx = metamatrix.rank - 1u - i;
            let dim_size = metamatrix.out_shape[dim_idx];

            let coord = temp_idx % dim_size;
            temp_idx = temp_idx / dim_size;

            offset_A += coord * metamatrix.strides_A[dim_idx];
            offset_B += coord * metamatrix.strides_B[dim_idx];
        }

        let val_A = A[offset_A];
        let val_B = B[offset_B];
        var res: f32 = 0.0;

        switch (metamatrix.op_type) {
            case 0u: { res = val_A + val_B; }
            case 1u: { res = val_A - val_B; }
            case 2u: { res = val_A * val_B; }
            case 3u: { res = val_A / val_B; }
            default: { res = val_A + val_B; }
        }

        // Write directly to offset_A (supports strided views of A)
        A[offset_A] = res;
        idx += stride;
    }
}