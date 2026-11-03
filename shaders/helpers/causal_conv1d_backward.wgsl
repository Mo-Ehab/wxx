// Backward pass of the fused depthwise causal 1-D convolution.
//
// A single dispatch covers three index ranges:
//   [0, B*D*L)              -> dX
//   [B*D*L, B*D*L + D*K)    -> dWeight (summed over batch and time)
//   [.., + D)               -> dBias   (summed over batch and time)
//
// dWeight/dBias are reduced in-kernel, so no atomics or partial buffers are
// needed.

struct CausalConvBwdMeta {
    batch: u32,
    channels: u32,
    length: u32,
    kernel: u32,
    total_dx: u32,
    total_dw: u32,
    compute_weight_grad: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> dY: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> dX: array<f32>;
@group(0) @binding(4) var<storage, read_write> dW: array<f32>;
@group(0) @binding(5) var<storage, read_write> dB: array<f32>;
@group(0) @binding(6) var<uniform> params: CausalConvBwdMeta;

// Strides of the saved input view.
struct XView {
    offset: u32,
    stride_b: u32,
    stride_c: u32,
    stride_l: u32,
};

@group(0) @binding(7) var<uniform> xview: XView;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total = params.total_dx
        + select(0u, params.total_dw, params.compute_weight_grad != 0u)
        + select(0u, params.channels, params.compute_weight_grad != 0u);
    let stride = num_workgroups.x * WORKGROUP_SIZE;

    var idx = global_id.x;
    while (idx < total) {
        if (idx < params.total_dx) {
            // dX[b, d, n] = sum_k weight[d, k] * dY[b, d, n + k]
            let n = idx % params.length;
            let rest = idx / params.length;
            let d = rest % params.channels;
            let b = rest / params.channels;

            var acc = 0.0;
            for (var k = 0u; k < params.kernel; k = k + 1u) {
                if (n + k < params.length) {
                    let src = ((b * params.channels + d) * params.length) + n + k;
                    acc = acc + W[d * params.kernel + k] * dY[src];
                }
            }
            dX[idx] = acc;
        } else if (idx < params.total_dx + params.total_dw) {
            // dW[d, k] = sum_{b, n >= k} dY[b, d, n] * X[b, d, n - k]
            let j = idx - params.total_dx;
            let k = j % params.kernel;
            let d = j / params.kernel;

            var acc = 0.0;
            for (var b = 0u; b < params.batch; b = b + 1u) {
                let row = (b * params.channels + d) * params.length;
                for (var n = k; n < params.length; n = n + 1u) {
                    let src = xview.offset
                        + b * xview.stride_b
                        + d * xview.stride_c
                        + (n - k) * xview.stride_l;
                    acc = acc + dY[row + n] * X[src];
                }
            }
            dW[j] = acc;
        } else {
            // dB[d] = sum_{b, n} dY[b, d, n]
            let d = idx - params.total_dx - params.total_dw;

            var acc = 0.0;
            for (var b = 0u; b < params.batch; b = b + 1u) {
                let row = (b * params.channels + d) * params.length;
                for (var n = 0u; n < params.length; n = n + 1u) {
                    acc = acc + dY[row + n];
                }
            }
            dB[d] = acc;
        }
        idx += stride;
    }
}
