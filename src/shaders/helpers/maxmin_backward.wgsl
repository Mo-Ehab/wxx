struct ReduceMeta {
    outer_size: u32,
    reduce_size: u32,
    inner_size: u32,
    numel: u32,
};

@group(0) @binding(0) var<storage, read> grad_output: array<f32>;
@group(0) @binding(1) var<storage, read> reduced_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(3) var<uniform> metamatrix: ReduceMeta;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let stride = num_workgroups.x * WORKGROUP_SIZE;
    var input_idx = global_id.x;

    while (input_idx < metamatrix.numel) {
        if (input_idx >= arrayLength(&grad_input)) { break; }

        let outer_idx = input_idx / (metamatrix.reduce_size * metamatrix.inner_size);
        let reduce_idx = (input_idx / metamatrix.inner_size) % metamatrix.reduce_size;
        let inner_idx = input_idx % metamatrix.inner_size;
        let output_idx = outer_idx * metamatrix.inner_size + inner_idx;

        if (reduced_indices[output_idx] == reduce_idx) {
            grad_input[input_idx] = grad_output[output_idx];
        } else {
            grad_input[input_idx] = 0.0;
        }
        input_idx += stride;
    }
}
