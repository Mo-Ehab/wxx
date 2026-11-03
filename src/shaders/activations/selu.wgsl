
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;

const ALPHA: f32 = 1.6732632423543772848170429916717;
const SCALE: f32 = 1.0507009873554804934193349852946;
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
        let x = X[i];
        if (x > 0.0) {
            Y[i] = SCALE * x;
        } else {
            Y[i] = SCALE * ALPHA * (exp(x) - 1.0);
        }
        i += stride;
    }
}