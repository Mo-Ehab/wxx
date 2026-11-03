

@group(0) @binding(0) var<storage, read> Y: array<f32>;  // Saved forward output Y
@group(0) @binding(1) var<storage, read_write> dX: array<f32>; // Output gradient


const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let stride = num_workgroups.x * WORKGROUP_SIZE;
    let total_elements = arrayLength(&Y);

    var i = global_id.x;
    while (i < total_elements) {
        if (i >= arrayLength(&Y)) { break; }
        let y_val = Y[i];
        dX[i] =  y_val * (1.0 - y_val);
        i += stride;
    }
}