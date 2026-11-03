// Fused single-workgroup real FFT (radix-2, decimation in time).
//
// One workgroup handles one batch row: it loads the real input in
// bit-reversed order into workgroup memory, runs every butterfly stage with
// workgroup barriers, and writes the N/2 + 1 non-redundant spectrum.
//
// This replaces the host-side loop of ``log2(N) + 2`` kernel launches with a
// single dispatch, which matters because the per-dispatch overhead dominates
// small transforms.

struct FFTPassMeta {
    batch_count: u32,
    n: u32,
    log2_n: u32,
    in_len: u32,
    out_len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: FFTPassMeta;

const MAX_N: u32 = 1024u;
const WG: u32 = 256u;
const PI: f32 = 3.14159265358979323846;

var<workgroup> buf: array<vec2<f32>, 1024>;

fn bit_reverse(val: u32, log2_n: u32) -> u32 {
    var x = val;
    var res = 0u;
    for (var i = 0u; i < log2_n; i = i + 1u) {
        res = (res << 1u) | (x & 1u);
        x >>= 1u;
    }
    return res;
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let n = params.n;
    let batch = wid.x;
    let tid = lid.x;

    // Load real input, zero-padding past ``in_len``, in bit-reversed order.
    for (var i = tid; i < n; i = i + WG) {
        var value = 0.0;
        if (i < params.in_len) {
            value = X[batch * params.in_len + i];
        }
        buf[bit_reverse(i, params.log2_n)] = vec2<f32>(value, 0.0);
    }
    workgroupBarrier();

    // In-place decimation-in-time butterflies.
    var m = 2u;
    for (var stage = 0u; stage < params.log2_n; stage = stage + 1u) {
        let half_m = m >> 1u;
        for (var p = tid; p < n / 2u; p = p + WG) {
            let k = p % half_m;
            let group = p / half_m;
            let i1 = group * m + k;
            let i2 = i1 + half_m;

            let angle = -2.0 * PI * f32(k) / f32(m);
            let tw = vec2<f32>(cos(angle), sin(angle));

            let u = buf[i1];
            let v = buf[i2];
            let t = vec2<f32>(v.x * tw.x - v.y * tw.y, v.x * tw.y + v.y * tw.x);

            buf[i1] = u + t;
            buf[i2] = u - t;
        }
        workgroupBarrier();
        m = m << 1u;
    }

    // Emit the non-redundant half of the spectrum.
    for (var j = tid; j < params.out_len; j = j + WG) {
        Y[batch * params.out_len + j] = buf[j];
    }
}
