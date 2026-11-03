struct BatchNormMeta {
    batch: u32,
    channels: u32,
    height: u32,
    width: u32,
    eps: f32,
    momentum: f32,
    is_training: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> dY: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> dgamma: array<f32>;
@group(0) @binding(4) var<storage, read> dbeta: array<f32>;
@group(0) @binding(5) var<storage, read> saved_mean: array<f32>;
@group(0) @binding(6) var<storage, read> inv_std: array<f32>;
@group(0) @binding(7) var<storage, read_write> dX: array<f32>;
@group(0) @binding(8) var<uniform> metamatrix: BatchNormMeta;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total_elements = metamatrix.batch * metamatrix.channels * metamatrix.height * metamatrix.width;
    let stride = num_workgroups.x * WORKGROUP_SIZE;
    let hw = metamatrix.height * metamatrix.width;
    let m_f32 = f32(metamatrix.batch * hw);

    var idx = global_id.x;
    while (idx < total_elements) {
        let c = (idx / hw) % metamatrix.channels;

        let mu = saved_mean[c];
        let inv_s = inv_std[c];
        let g = gamma[c];
        let dg = dgamma[c];
        let db = dbeta[c];

        let dy_val = dY[idx];
        let x_hat = (X[idx] - mu) * inv_s;

        // Exact analytical BatchNorm input gradient
        dX[idx] = (g * inv_s / m_f32) * (m_f32 * dy_val - db - x_hat * dg);

        idx += stride;
    }
}