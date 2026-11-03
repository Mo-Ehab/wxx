struct RMSNorm3DMeta {
    outer_size: u32,
    reduce_size: u32,
    inner_size: u32,
    eps: f32,
};

@group(0) @binding(0) var<storage, read> dY: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> inv_rms: array<f32>;
@group(0) @binding(4) var<storage, read_write> dX: array<f32>;
@group(0) @binding(5) var<uniform> metaMatrix: RMSNorm3DMeta;

const WORKGROUP_SIZE: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total_oi = metaMatrix.outer_size * metaMatrix.inner_size;
    let group_stride = num_workgroups.x;
    var out_idx = workgroup_id.x;

    while (out_idx < total_oi) {
        let o = out_idx / metaMatrix.inner_size;
        let i = out_idx % metaMatrix.inner_size;
        let r_val = inv_rms[out_idx];

        var thread_sum: f32 = 0.0;
        var r = local_id.x;
        while (r < metaMatrix.reduce_size) {
            let idx = o * (metaMatrix.reduce_size * metaMatrix.inner_size) + r * metaMatrix.inner_size + i;
            let dbar_x = dY[idx] * gamma[r];
            thread_sum += dbar_x * X[idx];
            r += WORKGROUP_SIZE;
        }

        sdata[local_id.x] = thread_sum;
        workgroupBarrier();

        for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
            if (local_id.x < s) {
                sdata[local_id.x] += sdata[local_id.x + s];
            }
            workgroupBarrier();
        }

        let c = (r_val * r_val / f32(metaMatrix.reduce_size)) * sdata[0];

        r = local_id.x;
        while (r < metaMatrix.reduce_size) {
            let idx = o * (metaMatrix.reduce_size * metaMatrix.inner_size) + r * metaMatrix.inner_size + i;
            let dbar_x = dY[idx] * gamma[r];
            let x_val = X[idx];
            dX[idx] = r_val * (dbar_x - x_val * c);
            r += WORKGROUP_SIZE;
        }

        out_idx += group_stride;
    }
}