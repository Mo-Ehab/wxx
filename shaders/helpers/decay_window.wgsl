// Causal exponential decay window: W[d, t] = exp(-|decay[stage, d]| * t).
//
// Fusing this replaces the abs -> mul -> negate -> exp chain (and its
// backward graph) with a single dispatch.

struct DecayWindowMeta {
    rows: u32,      // decay.shape[0] (number of stages)
    channels: u32,  // decay.shape[1]
    length: u32,    // number of time steps
    stage: u32,
};

@group(0) @binding(0) var<storage, read> Decay: array<f32>;
@group(0) @binding(1) var<storage, read_write> W: array<f32>;
@group(0) @binding(2) var<uniform> params: DecayWindowMeta;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total = params.channels * params.length;
    let stride = num_workgroups.x * WORKGROUP_SIZE;

    var idx = global_id.x;
    while (idx < total) {
        let t = idx % params.length;
        let d = idx / params.length;
        let rate = Decay[params.stage * params.channels + d];
        W[idx] = exp(-abs(rate) * f32(t));
        idx += stride;
    }
}
