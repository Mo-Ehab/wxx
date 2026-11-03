// One-sided complex spectrum -> full Hermitian spectrum, written in
// bit-reversed order so that the DIT butterfly stages can run in place.

struct BitRevComplexMeta {
    batch_count: u32,
    n: u32,
    log2_n: u32,
    in_len: u32,
};

@group(0) @binding(0) var<storage, read> Y: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> Z: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: BitRevComplexMeta;

const WORKGROUP_SIZE: u32 = 256u;

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
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total = params.batch_count * params.n;
    let stride = num_workgroups.x * WORKGROUP_SIZE;
    let nyquist = params.n / 2u;

    var idx = global_id.x;
    while (idx < total) {
        let batch = idx / params.n;
        let i = idx % params.n;

        var value = vec2<f32>(0.0, 0.0);
        if (i <= nyquist) {
            if (i < params.in_len) {
                value = Y[batch * params.in_len + i];
            }
        } else {
            // Hermitian symmetry: Y[N - k] = conj(Y[k]).
            let mirror = params.n - i;
            if (mirror < params.in_len) {
                let m = Y[batch * params.in_len + mirror];
                value = vec2<f32>(m.x, -m.y);
            }
        }

        let rev = bit_reverse(i, params.log2_n);
        Z[batch * params.n + rev] = value;
        idx += stride;
    }
}
