struct RMSNorm3DMeta {
    outer_size: u32,
    reduce_size: u32,
    inner_size: u32,
    eps: f32,
};

@group(0) @binding(0) var<storage, read> dY: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> inv_rms: array<f32>;
@group(0) @binding(3) var<storage, read_write> dgamma: array<f32>;
@group(0) @binding(4) var<uniform> metaMatrix: RMSNorm3DMeta;

const WORKGROUP_SIZE: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let inner_sz = max(metaMatrix.inner_size, 1u);
    let total_oi = metaMatrix.outer_size * inner_sz;
    let feature_stride = num_workgroups.x;
    var r = workgroup_id.x;

    while (r < metaMatrix.reduce_size) {
        var thread_sum = 0.0;
        var idx = local_id.x;

        while (idx < total_oi) {
            let o = idx / inner_sz;
            let i = idx % inner_sz;
            let physical_idx = o * (metaMatrix.reduce_size * inner_sz) + r * inner_sz + i;
            thread_sum += dY[physical_idx] * X[physical_idx] * inv_rms[idx];
            idx += WORKGROUP_SIZE;
        }

        sdata[local_id.x] = thread_sum;
        workgroupBarrier();

        for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
            if (local_id.x < s) {
                sdata[local_id.x] += sdata[local_id.x + s];
            }
            workgroupBarrier();
        }

        if (local_id.x == 0u) {
            dgamma[r] = sdata[0];
        }

        r += feature_stride;
    }
}