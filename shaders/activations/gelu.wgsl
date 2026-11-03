@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;

const SQRT_2_OVER_PI: f32 = 0.7978845608;
const COEFF: f32 = 0.044715;
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
        if (x > 10.0) {
            Y[i] = x;
        } else if (x < -10.0) {
            Y[i] = 0;
        } else{
        let u = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
        Y[i] = 0.5 * x * (1.0 + tanh(u));
        }
        i += stride;
    }
}