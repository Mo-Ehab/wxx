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
};

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> metamatrix: ConvMeta;

const TILE_SIZE: u32 = 16u;
const MAX_ROW_WORKGROUPS: u32 = 65535u;

var<workgroup> tile_X: array<array<f32, 17>, 16>;
var<workgroup> tile_W: array<array<f32, 17>, 16>;

fn load_im2col(row: u32, reduction_idx: u32) -> f32 {
    let output_spatial = metamatrix.out_h * metamatrix.out_w;
    let batch_idx = row / output_spatial;
    let output_position = row % output_spatial;
    let output_h = output_position / metamatrix.out_w;
    let output_w = output_position % metamatrix.out_w;

    let kernel_area = metamatrix.kh * metamatrix.kw;
    let input_c = reduction_idx / kernel_area;
    let kernel_position = reduction_idx % kernel_area;
    let kernel_h = kernel_position / metamatrix.kw;
    let kernel_w = kernel_position % metamatrix.kw;

    let input_h = i32(output_h * metamatrix.stride_h + kernel_h * metamatrix.dilation_h) -
        i32(metamatrix.pad_h);
    let input_w = i32(output_w * metamatrix.stride_w + kernel_w * metamatrix.dilation_w) -
        i32(metamatrix.pad_w);

    if (input_h < 0 || input_h >= i32(metamatrix.in_h) ||
        input_w < 0 || input_w >= i32(metamatrix.in_w)) {
        return 0.0;
    }

    let input_idx = ((batch_idx * metamatrix.in_c + input_c) * metamatrix.in_h +
        u32(input_h)) * metamatrix.in_w + u32(input_w);
    return X[input_idx];
}

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let output_rows = metamatrix.batch * metamatrix.out_h * metamatrix.out_w;
    let reduction_size = metamatrix.in_c * metamatrix.kh * metamatrix.kw;
    // X lanes follow contiguous NCHW output positions; Y lanes select channels.
    let row = (workgroup_id.z * MAX_ROW_WORKGROUPS + workgroup_id.x) *
        TILE_SIZE + local_id.x;
    let output_c = workgroup_id.y * TILE_SIZE + local_id.y;

    var accumulator = 0.0;
    var reduction_base = 0u;

    while (reduction_base < reduction_size) {
        let input_k = reduction_base + local_id.y;
        if (row < output_rows && input_k < reduction_size) {
            tile_X[local_id.x][local_id.y] = load_im2col(row, input_k);
        } else {
            tile_X[local_id.x][local_id.y] = 0.0;
        }

        let weight_k = reduction_base + local_id.x;
        if (output_c < metamatrix.out_c && weight_k < reduction_size) {
            tile_W[local_id.x][local_id.y] =
                W[output_c * reduction_size + weight_k];
        } else {
            tile_W[local_id.x][local_id.y] = 0.0;
        }

        workgroupBarrier();

        for (var k = 0u; k < TILE_SIZE; k++) {
            accumulator = fma(
                tile_X[local_id.x][k],
                tile_W[k][local_id.y],
                accumulator
            );
        }

        workgroupBarrier();
        reduction_base += TILE_SIZE;
    }

    if (row < output_rows && output_c < metamatrix.out_c) {
        let batch_idx = row / (metamatrix.out_h * metamatrix.out_w);
        let output_position = row % (metamatrix.out_h * metamatrix.out_w);
        var value = accumulator;
        if (metamatrix.has_bias == 1u) {
            value += bias[output_c];
        }

        let output_idx = (batch_idx * metamatrix.out_c + output_c) *
            (metamatrix.out_h * metamatrix.out_w) + output_position;
        Y[output_idx] = value;
    }
}
