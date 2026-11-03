struct StridedSoftmaxMeta {
    numel: u32,
    rank: u32,
    axis: u32,
    _pad0: u32,
    shape: array<vec4<u32>, 2>,   // Supports up to 8D shapes
    strides: array<vec4<u32>, 2>, // Dynamic input/output strides
};

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<storage, read> metamatrix: StridedSoftmaxMeta;

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
        if (outer_idx >= arrayLength(&Y)) { break; }
        // --- Step 1: Decompose outer_idx into baseline physical VRAM offset ---
        var temp = outer_idx;
        var base_offset: u32 = 0u;

        // Iterate through all dimensions EXCEPT target reduction axis
        for (var d = 0u; d < metamatrix.rank; d = d + 1u) {
            if (d == metamatrix.axis) {
                continue;
            }

            let dim_size = get_shape(d);
            let coord = temp % dim_size;
            temp = temp / dim_size;

            base_offset += coord * get_stride(d);
        }

        // --- Step 2: Pass 1 — Find Row Maximum for Numerical Stability ---
        var max_val: f32 = -1e38;
        for (var c = 0u; c < axis_dim; c = c + 1u) {
            let phys_idx = base_offset + c * axis_stride;
            max_val = max(max_val, X[phys_idx]);
        }

        // --- Step 3: Pass 2 — Compute Sum of Exponentials ---
        var sum_exp: f32 = 0.0;
        for (var c = 0u; c < axis_dim; c = c + 1u) {
            let phys_idx = base_offset + c * axis_stride;
            sum_exp += exp(X[phys_idx] - max_val);
        }

        // --- Step 4: Pass 3 — Normalize and Write Probabilities Y ---
        let inv_sum = 1.0 / sum_exp;
        for (var c = 0u; c < axis_dim; c = c + 1u) {
            let phys_idx = base_offset + c * axis_stride;
            Y[phys_idx] = exp(X[phys_idx] - max_val) * inv_sum;
        }

        outer_idx += stride_grid;
    }
}