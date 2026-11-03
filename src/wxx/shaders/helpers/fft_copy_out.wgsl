// Copy the non-redundant part (N/2 + 1) of a full complex spectrum into a
// contiguous output buffer.

struct FFTCopyMeta {
    batch_count: u32,
    n: u32,
    out_len: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> Z: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> Y: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: FFTCopyMeta;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total = params.batch_count * params.out_len;
    let stride = num_workgroups.x * WORKGROUP_SIZE;

    var idx = global_id.x;
    while (idx < total) {
        let batch = idx / params.out_len;
        let j = idx % params.out_len;
        Y[idx] = Z[batch * params.n + j];
        idx += stride;
    }
}
