// Backward pass of elementwise multiplication, producing both input
// gradients in one dispatch:
//
//   dA = grad * B
//   dB = grad * A
//
// This replaces two separate pointwise kernels whenever both operands need a
// gradient (the common case for weight/activation products).

struct MulBwdMeta {
    numel: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> G: array<f32>;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> dA: array<f32>;
@group(0) @binding(4) var<storage, read_write> dB: array<f32>;
@group(0) @binding(5) var<uniform> params: MulBwdMeta;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let stride = num_workgroups.x * WORKGROUP_SIZE;

    var idx = global_id.x;
    while (idx < params.numel) {
        let g = G[idx];
        dA[idx] = g * B[idx];
        dB[idx] = g * A[idx];
        idx += stride;
    }
}
