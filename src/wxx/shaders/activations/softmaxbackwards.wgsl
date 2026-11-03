struct StridedSoftmaxMeta {
    numel: u32,
    rank: u32,
    axis: u32,
    _pad0: u32,
    shape: array<vec4<u32>, 2>,
    strides: array<vec4<u32>, 2>,
};

@group(0) @binding(0) var<storage, read> dY: array<f32>;
@group(0) @binding(1) var<storage, read> Y: array<f32>;
@group(0) @binding(2) var<storage, read_write> dX: array<f32>;
@group(0) @binding(3) var<storage, read> metamatrix: StridedSoftmaxMeta;

fn get_shape(dim: u32) -> u32 {
    return metamatrix.shape[dim / 4u][dim % 4u];
}

fn get_stride(dim: u32) -> u32 {
    return metamatrix.strides[dim / 4u][dim % 4u];
}

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let axis_dim = get_shape(metamatrix.axis);
    let axis_stride = get_stride(metamatrix.axis);
    let total_outer = metamatrix.numel / axis_dim;

    let stride_grid = num_workgroups.x * WORKGROUP_SIZE;
    var outer_idx = global_id.x;

    while (outer_idx < total_outer) {
        if (outer_idx >= arrayLength(&dX)) { break; }
        var temp = outer_idx;
        var base_offset: u32 = 0u;

        for (var d = 0u; d < metamatrix.rank; d = d + 1u) {
            if (d == metamatrix.axis) {
                continue;
            }
            let dim_size = get_shape(d);
            let coord = temp % dim_size;
            temp = temp / dim_size;

            base_offset += coord * get_stride(d);
        }

        // Pass 1: Compute Row Dot Product S = sum(dY * Y) along reduction axis
        var dot_product: f32 = 0.0;
        for (var c = 0u; c < axis_dim; c = c + 1u) {
            let phys_idx = base_offset + c * axis_stride;
            dot_product += dY[phys_idx] * Y[phys_idx];
        }

        // Pass 2: Write Gradient dX_i = Y_i * (dY_i - S)
        for (var c = 0u; c < axis_dim; c = c + 1u) {
            let phys_idx = base_offset + c * axis_stride;
            let y_val = Y[phys_idx];
            let dy_val = dY[phys_idx];
            dX[phys_idx] = y_val * (dy_val - dot_product);
        }

        outer_idx += stride_grid;
    }
}