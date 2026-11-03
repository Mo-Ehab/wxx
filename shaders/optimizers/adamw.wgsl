struct AdamWMeta {
    numel: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    bias_correction1: f32, // Precomputed on CPU (1.0 - beta1^t)
    bias_correction2: f32, // Precomputed on CPU (1.0 - beta2^t)
};

@group(0) @binding(0) var<storage, read_write> param: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> exp_avg: array<f32>;
@group(0) @binding(3) var<storage, read_write> exp_avg_sq: array<f32>;
@group(0) @binding(4) var<storage, read> metamatrix: AdamWMeta;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let stride = num_workgroups.x * WORKGROUP_SIZE;
    var idx = global_id.x;

    let bc1 = metamatrix.bias_correction1;
    let bc2 = metamatrix.bias_correction2;
    let beta1 = metamatrix.beta1;
    let beta2 = metamatrix.beta2;
    let lr = metamatrix.lr;
    let wd = metamatrix.weight_decay;

    while (idx < metamatrix.numel) {
        let p = param[idx];
        let g = grad[idx];

        let m = beta1 * exp_avg[idx] + (1.0 - beta1) * g;
        let v = beta2 * exp_avg_sq[idx] + (1.0 - beta2) * g * g;

        exp_avg[idx] = m;
        exp_avg_sq[idx] = v;

        // 2. Bias correction
        let m_hat = m / bc1;
        let v_hat = v / bc2;

        // 3. Adaptive step & decoupled weight decay
        let adaptive_step = m_hat / (sqrt(v_hat) + metamatrix.eps);
        let decay_step = wd * p;

        // 4. Update parameter buffer directly
        param[idx] = p - lr * (adaptive_step + decay_step);

        idx += stride;
    }
}