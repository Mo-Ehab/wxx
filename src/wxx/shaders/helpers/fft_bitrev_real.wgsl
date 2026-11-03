// Real -> complex bit-reversal stage used by the radix-2 FFT.
//
// Each workgroup invocation handles one index of the full zero-padded
// sequence.  Input samples beyond `in_len` are treated as zeros, which lets
// rfft(x, n=fft_size) zero-pad (or truncate) without an extra kernel.

struct BitRevRealMeta {
    batch_count: u32,
    n: u32,
    log2_n: u32,
    in_len: u32,
};

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Z: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: BitRevRealMeta;

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

    var idx = global_id.x;
    while (idx < total) {
        let batch = idx / params.n;
        let i = idx % params.n;
        let rev = bit_reverse(i, params.log2_n);

        var value = 0.0;
        if (i < params.in_len) {
            value = X[batch * params.in_len + i];
        }

        Z[batch * params.n + rev] = vec2<f32>(value, 0.0);
        idx += stride;
    }
}
