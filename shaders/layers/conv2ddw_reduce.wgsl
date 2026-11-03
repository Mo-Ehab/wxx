struct ReduceMeta {
    total_weights: u32,
    split_count: u32,
};

@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> dW: array<f32>;
@group(0) @binding(2) var<storage, read> metamatrix: ReduceMeta;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let grid_stride = num_workgroups.x * 256u;
    for (var weight_idx = global_id.x;
        weight_idx < metamatrix.total_weights;
        weight_idx += grid_stride) {
        var accumulator = 0.0;
        for (var split_idx = 0u; split_idx < metamatrix.split_count; split_idx++) {
            accumulator += partials[split_idx * metamatrix.total_weights + weight_idx];
        }
        dW[weight_idx] = accumulator;
    }
}
