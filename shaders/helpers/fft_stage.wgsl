// One radix-2 Cooley-Tukey butterfly stage, operating in place on a
// bit-reversed complex buffer.  Pairs written by different invocations are
// disjoint, so no global barrier is needed inside a stage.
//
// The host launches this kernel once per stage; the dispatches are ordered,
// which provides the synchronisation between consecutive stages.

struct FFTStageMeta {
    batch_count: u32,
    n: u32,
    log2_n: u32,
    stage: u32,
    inverse: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> Z: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: FFTStageMeta;

const WORKGROUP_SIZE: u32 = 256u;
const PI: f32 = 3.14159265358979323846;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let half_n = params.n / 2u;
    let total_pairs = params.batch_count * half_n;
    let stride = num_workgroups.x * WORKGROUP_SIZE;

    let m = 1u << (params.stage + 1u);
    let half_m = m >> 1u;
    let sign = select(-1.0, 1.0, params.inverse != 0u);

    var idx = global_id.x;
    while (idx < total_pairs) {
        let batch = idx / half_n;
        let pair = idx % half_n;

        let k = pair % half_m;
        let group = pair / half_m;
        let i1 = group * m + k;
        let i2 = i1 + half_m;

        let angle = sign * 2.0 * PI * f32(k) / f32(m);
        let twiddle = vec2<f32>(cos(angle), sin(angle));

        let base = batch * params.n;
        let u = Z[base + i1];
        let v = complex_mul(Z[base + i2], twiddle);

        Z[base + i1] = u + v;
        Z[base + i2] = u - v;

        idx += stride;
    }
}
