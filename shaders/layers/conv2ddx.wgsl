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

@group(0) @binding(0) var<storage, read> dY: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read_write> dX: array<f32>;
@group(0) @binding(3) var<storage, read> metamatrix: ConvMeta;

const TILE_SIZE: u32 = 16u;
const MAX_ROW_WORKGROUPS: u32 = 65535u;

var<workgroup> tile_dY: array<array<f32, 17>, 16>;
var<workgroup> tile_W: array<array<f32, 17>, 16>;

fn load_output_gradient(row: u32, reduction_idx: u32) -> f32 {
    let input_spatial = metamatrix.in_h * metamatrix.in_w;
    let batch_idx = row / input_spatial;
    let input_position = row % input_spatial;
    let input_h = input_position / metamatrix.in_w;
    let input_w = input_position % metamatrix.in_w;

    let kernel_area = metamatrix.kh * metamatrix.kw;
    let output_c = reduction_idx / kernel_area;
    let kernel_position = reduction_idx % kernel_area;
    let kernel_h = kernel_position / metamatrix.kw;
    let kernel_w = kernel_position % metamatrix.kw;

    let output_h_shifted = i32(input_h + metamatrix.pad_h) -
        i32(kernel_h * metamatrix.dilation_h);
    let output_w_shifted = i32(input_w + metamatrix.pad_w) -
        i32(kernel_w * metamatrix.dilation_w);

    if (output_h_shifted < 0 ||
        output_h_shifted % i32(metamatrix.stride_h) != 0 ||
        output_w_shifted < 0 ||
        output_w_shifted % i32(metamatrix.stride_w) != 0) {
        return 0.0;
    }

    let output_h = u32(output_h_shifted / i32(metamatrix.stride_h));
    let output_w = u32(output_w_shifted / i32(metamatrix.stride_w));
    if (output_h >= metamatrix.out_h || output_w >= metamatrix.out_w) {
        return 0.0;
    }

    let output_spatial = metamatrix.out_h * metamatrix.out_w;
    let output_idx = (batch_idx * metamatrix.out_c + output_c) * output_spatial +
        output_h * metamatrix.out_w + output_w;
    return dY[output_idx];
}

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let output_rows = metamatrix.batch * metamatrix.in_h * metamatrix.in_w;
    let reduction_size = metamatrix.out_c * metamatrix.kh * metamatrix.kw;
    let row = (workgroup_id.z * MAX_ROW_WORKGROUPS + workgroup_id.x) *
        TILE_SIZE + local_id.x;
    let input_c = workgroup_id.y * TILE_SIZE + local_id.y;

    var accumulator = 0.0;
    var reduction_base = 0u;

    while (reduction_base < reduction_size) {
        let gradient_k = reduction_base + local_id.y;
        if (row < output_rows && gradient_k < reduction_size) {
            tile_dY[local_id.x][local_id.y] =
                load_output_gradient(row, gradient_k);
        } else {
            tile_dY[local_id.x][local_id.y] = 0.0;
        }

        let weight_k = reduction_base + local_id.x;
        if (input_c < metamatrix.in_c && weight_k < reduction_size) {
            let kernel_area = metamatrix.kh * metamatrix.kw;
            let output_c = weight_k / kernel_area;
            let kernel_position = weight_k % kernel_area;
            let weight_idx = ((output_c * metamatrix.in_c + input_c) *
                metamatrix.kh + kernel_position / metamatrix.kw) *
                metamatrix.kw + kernel_position % metamatrix.kw;
            tile_W[local_id.x][local_id.y] = W[weight_idx];
        } else {
            tile_W[local_id.x][local_id.y] = 0.0;
        }

        workgroupBarrier();

        for (var k = 0u; k < TILE_SIZE; k++) {
            accumulator = fma(
                tile_dY[local_id.x][k],
                tile_W[k][local_id.y],
                accumulator
            );
        }

        workgroupBarrier();
        reduction_base += TILE_SIZE;
    }

    if (row < output_rows && input_c < metamatrix.in_c) {
        let batch_idx = row / (metamatrix.in_h * metamatrix.in_w);
        let input_position = row % (metamatrix.in_h * metamatrix.in_w);
        let input_idx = (batch_idx * metamatrix.in_c + input_c) *
            (metamatrix.in_h * metamatrix.in_w) + input_position;
        dX[input_idx] = accumulator;
    }
}
