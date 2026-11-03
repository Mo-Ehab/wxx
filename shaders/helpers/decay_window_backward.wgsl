// Backward pass of the causal exponential decay window.
//
//   W[d, t]  = exp(-|r_d| * t)
//   dW/dr_d  = sum_t grad[d, t] * W[d, t] * (-sign(r_d) * t)
//
// The output has the full (rows, channels) shape of the decay parameter, with
// zeros in every row except ``stage`` -- so the gradient scatter is fused in
// and no separate narrow/assign kernel is needed.

struct DecayWindowBwdMeta {
    rows: u32,
    channels: u32,
    length: u32,
    stage: u32,
};

@group(0) @binding(0) var<storage, read> Grad: array<f32>;
@group(0) @binding(1) var<storage, read> Decay: array<f32>;
@group(0) @binding(2) var<storage, read_write> dDecay: array<f32>;
@group(0) @binding(3) var<uniform> params: DecayWindowBwdMeta;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total = params.rows * params.channels;
    let stride = num_workgroups.x * WORKGROUP_SIZE;

    var idx = global_id.x;
    while (idx < total) {
        let d = idx % params.channels;
        let row = idx / params.channels;

        var acc = 0.0;
        if (row == params.stage) {
            let rate = Decay[params.stage * params.channels + d];
            let sign_rate = sign(rate);
            for (var t = 0u; t < params.length; t = t + 1u) {
                let w = exp(-abs(rate) * f32(t));
                acc = acc + Grad[d * params.length + t] * w * (-sign_rate * f32(t));
            }
        }
        dDecay[idx] = acc;
        idx += stride;
    }
}
