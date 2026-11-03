// Fused depthwise causal 1-D convolution.
//
//   out[b, d, n] = bias[d] + sum_k weight[d, k] * x[b, d, n - k]
//
// The input may be a strided view (``offset``/``stride_*``), which lets the
// operator consume the ``(B, L, D) -> (B, D, L)`` transpose without a
// ``contiguous()`` copy.  One dispatch replaces the concat + per-tap mul/add
// chain used by the generic tensor ops.

struct CausalConvMeta {
    batch: u32,
    channels: u32,
    length: u32,
    kernel: u32,
    offset: u32,
    stride_b: u32,
    stride_c: u32,
    stride_l: u32,
};

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> params: CausalConvMeta;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total = params.batch * params.channels * params.length;
    let stride = num_workgroups.x * WORKGROUP_SIZE;

    var idx = global_id.x;
    while (idx < total) {
        let n = idx % params.length;
        let rest = idx / params.length;
        let d = rest % params.channels;
        let b = rest / params.channels;

        var acc = Bias[d];
        for (var k = 0u; k < params.kernel; k = k + 1u) {
            if (n >= k) {
                let src = params.offset
                    + b * params.stride_b
                    + d * params.stride_c
                    + (n - k) * params.stride_l;
                acc = acc + W[d * params.kernel + k] * X[src];
            }
        }
        Y[idx] = acc;
        idx += stride;
    }
}
