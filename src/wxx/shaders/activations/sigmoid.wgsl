

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;

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
        // Clamp input to avoid floating point overflow in exp()
        let x_clamped = clamp(X[i], -88.0, 88.0);
        Y[i] = 1.0 / (1.0 + exp(-x_clamped));
        i += stride;
    }
}