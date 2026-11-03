// Extract the real part of a full inverse-FFT buffer and normalise by 1/N.

struct IFFTWriteMeta {
    batch_count: u32,
    n: u32,
    inv_n: f32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> Z: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> X: array<f32>;
@group(0) @binding(2) var<uniform> params: IFFTWriteMeta;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total = params.batch_count * params.n;
    let stride = num_workgroups.x * WORKGROUP_SIZE;

    var idx = global_id.x;
    while (idx < total) {
        X[idx] = Z[idx].x * params.inv_n;
        idx += stride;
    }
}
