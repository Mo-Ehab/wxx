// Fused single-workgroup inverse real FFT (radix-2, decimation in time).
//
// One workgroup handles one batch row: it expands the one-sided spectrum into
// the full Hermitian spectrum in bit-reversed order, runs every butterfly
// stage with workgroup barriers, and writes the real result scaled by 1/N.

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

@group(0) @binding(0) var<storage, read> Y: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> X: array<f32>;
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
    let nyquist = n / 2u;

    // Expand the one-sided spectrum with Hermitian symmetry.
    for (var i = tid; i < n; i = i + WG) {
        var value = vec2<f32>(0.0, 0.0);
        if (i <= nyquist) {
            if (i < params.in_len) {
                value = Y[batch * params.in_len + i];
            }
        } else {
            let mirror = n - i;
            if (mirror < params.in_len) {
                let m = Y[batch * params.in_len + mirror];
                value = vec2<f32>(m.x, -m.y);
            }
        }
        buf[bit_reverse(i, params.log2_n)] = value;
    }
    workgroupBarrier();

    var m = 2u;
    for (var stage = 0u; stage < params.log2_n; stage = stage + 1u) {
        let half_m = m >> 1u;
        for (var p = tid; p < n / 2u; p = p + WG) {
            let k = p % half_m;
            let group = p / half_m;
            let i1 = group * m + k;
            let i2 = i1 + half_m;

            let angle = 2.0 * PI * f32(k) / f32(m);
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

    let inv_n = 1.0 / f32(n);
    for (var j = tid; j < n; j = j + WG) {
        X[batch * n + j] = buf[j].x * inv_n;
    }
}
