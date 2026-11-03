struct ConvMeta {
    batch: u32,
    in_c: u32,
    in_h: u32,
    in_w: u32,
    out_c: u32,
    out_h: u32,
    out_w: u32,
    kh: u32,
    kw: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    has_bias: u32,
    split_count: u32,
    split_size: u32,
};

@group(0) @binding(0) var<storage, read> dY: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> dW: array<f32>;
@group(0) @binding(3) var<storage, read> metamatrix: ConvMeta;

const TILE_SIZE: u32 = 16u;
const MAX_FILTER_WORKGROUPS: u32 = 65535u;

var<workgroup> tile_dY: array<array<f32, 17>, 16>;
var<workgroup> tile_X: array<array<f32, 17>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let kernel_area = metamatrix.kh * metamatrix.kw;
    let filter_size = metamatrix.in_c * kernel_area;
    // Flatten input-channel and kernel coordinates so adjacent lanes write adjacent dW values.
    let split_idx = workgroup_id.z % metamatrix.split_count;
    let filter_group_z = workgroup_id.z / metamatrix.split_count;
    let filter_idx = (filter_group_z * MAX_FILTER_WORKGROUPS + workgroup_id.x) *
        TILE_SIZE + local_id.x;
    let output_c = workgroup_id.y * TILE_SIZE + local_id.y;

    let input_c = filter_idx / kernel_area;
    let kernel_position = filter_idx % kernel_area;
    let kernel_h = kernel_position / metamatrix.kw;
    let kernel_w = kernel_position % metamatrix.kw;

    let total_spatial = metamatrix.batch * metamatrix.out_h * metamatrix.out_w;
    let output_spatial = metamatrix.out_h * metamatrix.out_w;
    let input_spatial = metamatrix.in_h * metamatrix.in_w;

    var accumulator = 0.0;
    let segment_start = split_idx * metamatrix.split_size;
    let segment_end = min(segment_start + metamatrix.split_size, total_spatial);
    var spatial_base = segment_start;

    while (spatial_base < segment_end) {
        let dy_spatial = spatial_base + local_id.x;
        if (output_c < metamatrix.out_c && dy_spatial < segment_end) {
            let batch_idx = dy_spatial / output_spatial;
            let output_position = dy_spatial % output_spatial;
            let dy_idx = (batch_idx * metamatrix.out_c + output_c) *
                output_spatial + output_position;
            tile_dY[local_id.y][local_id.x] = dY[dy_idx];
        } else {
            tile_dY[local_id.y][local_id.x] = 0.0;
        }

        let x_spatial = spatial_base + local_id.y;
        if (filter_idx < filter_size && x_spatial < segment_end) {
            let batch_idx = x_spatial / output_spatial;
            let output_position = x_spatial % output_spatial;
            let output_h = output_position / metamatrix.out_w;
            let output_w = output_position % metamatrix.out_w;
            let input_h = i32(output_h * metamatrix.stride_h +
                kernel_h * metamatrix.dilation_h) - i32(metamatrix.pad_h);
            let input_w = i32(output_w * metamatrix.stride_w +
                kernel_w * metamatrix.dilation_w) - i32(metamatrix.pad_w);

            if (input_h >= 0 && input_h < i32(metamatrix.in_h) &&
                input_w >= 0 && input_w < i32(metamatrix.in_w)) {
                let x_idx = (batch_idx * metamatrix.in_c + input_c) *
                    input_spatial + u32(input_h) * metamatrix.in_w + u32(input_w);
                tile_X[local_id.y][local_id.x] = X[x_idx];
            } else {
                tile_X[local_id.y][local_id.x] = 0.0;
            }
        } else {
            tile_X[local_id.y][local_id.x] = 0.0;
        }

        workgroupBarrier();

        for (var k = 0u; k < TILE_SIZE; k++) {
            accumulator = fma(
                tile_dY[local_id.y][k],
                tile_X[k][local_id.x],
                accumulator
            );
        }

        workgroupBarrier();
        spatial_base += TILE_SIZE;
    }

    if (filter_idx < filter_size && output_c < metamatrix.out_c) {
        let weight_idx = output_c * filter_size + filter_idx;
        if (metamatrix.split_count == 1u) {
            dW[weight_idx] = accumulator;
        } else {
            let total_weights = metamatrix.out_c * filter_size;
            dW[split_idx * total_weights + weight_idx] = accumulator;
        }
    }
}
