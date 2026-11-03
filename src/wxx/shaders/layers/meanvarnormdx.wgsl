struct Norm3DMeta {
    outer_size: u32,
    reduce_size: u32,
    inner_size: u32,
    eps: f32,
};

@group(0) @binding(0) var<storage, read> dY: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> saved_mean: array<f32>;
@group(0) @binding(4) var<storage, read> inv_std: array<f32>;
@group(0) @binding(5) var<storage, read_write> dX: array<f32>;
@group(0) @binding(6) var<uniform> metaMatrix: Norm3DMeta;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> sdata_c1: array<f32, 256>;
var<workgroup> sdata_c2: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let o = workgroup_id.y;
    if (o >= metaMatrix.outer_size) { return; }

    var i = workgroup_id.x;
    let inner_stride = num_workgroups.x;

    while (i < metaMatrix.inner_size) {
        let out_idx = o * metaMatrix.inner_size + i;
        let mu_val = saved_mean[out_idx];
        let inv_s = inv_std[out_idx];
        let base_offset = o * (metaMatrix.reduce_size * metaMatrix.inner_size) + i;

        // 1. Grid-stride accumulation over reduce_size
        var thread_sum_c1: f32 = 0.0;
        var thread_sum_c2: f32 = 0.0;
        var r = local_id.x;

        while (r < metaMatrix.reduce_size) {
            let physical_idx = base_offset + r * metaMatrix.inner_size;
            let dy_val = dY[physical_idx];
            let x_val = X[physical_idx];
            let g_val = gamma[r];

            let dbar_x = dy_val * g_val;
            let x_hat = (x_val - mu_val) * inv_s;

            thread_sum_c1 += dbar_x;
            thread_sum_c2 += dbar_x * x_hat;

            r += WORKGROUP_SIZE;
        }

        sdata_c1[local_id.x] = thread_sum_c1;
        sdata_c2[local_id.x] = thread_sum_c2;
        workgroupBarrier();

        // 2. Parallel tree reduction
        for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
            if (local_id.x < s) {
                sdata_c1[local_id.x] += sdata_c1[local_id.x + s];
                sdata_c2[local_id.x] += sdata_c2[local_id.x + s];
            }
            workgroupBarrier();
        }

        let c1_mean = sdata_c1[0] / f32(metaMatrix.reduce_size);
        let c2_mean = sdata_c2[0] / f32(metaMatrix.reduce_size);

        // 3. Compute and write dX
        r = local_id.x;
        while (r < metaMatrix.reduce_size) {
            let physical_idx = base_offset + r * metaMatrix.inner_size;
            let dy_val = dY[physical_idx];
            let x_val = X[physical_idx];
            let g_val = gamma[r];

            let dbar_x = dy_val * g_val;
            let x_hat = (x_val - mu_val) * inv_s;

            dX[physical_idx] = inv_s * (dbar_x - c1_mean - x_hat * c2_mean);

            r += WORKGROUP_SIZE;
        }

        i += inner_stride;
    }
}