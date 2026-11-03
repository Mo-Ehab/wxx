struct Norm3DMeta {
    outer_size: u32,
    reduce_size: u32,
    inner_size: u32,
    eps: f32,
};

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> saved_mean: array<f32>;
@group(0) @binding(5) var<storage, read_write> inv_std: array<f32>;
@group(0) @binding(6) var<uniform> metaMatrix: Norm3DMeta;

const WORKGROUP_SIZE: u32 = 256u;

// Dual shared memory arrays for single-pass reduction
var<workgroup> sdata_sum: array<f32, 256>;
var<workgroup> sdata_sq_sum: array<f32, 256>;

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

        // =========================================================================
        // SINGLE COMBINED REDUCTION PASS: Sum(X) and Sum(X^2)
        // =========================================================================
        var thread_sum: f32 = 0.0;
        var thread_sq_sum: f32 = 0.0;
        var r = local_id.x;

        let base_offset = o * (metaMatrix.reduce_size * metaMatrix.inner_size) + i;

        while (r < metaMatrix.reduce_size) {
            let physical_idx = base_offset + r * metaMatrix.inner_size;
            let x_val = X[physical_idx];

            thread_sum += x_val;
            thread_sq_sum += x_val * x_val;

            r += WORKGROUP_SIZE;
        }

        sdata_sum[local_id.x] = thread_sum;
        sdata_sq_sum[local_id.x] = thread_sq_sum;
        workgroupBarrier();

        // Parallel Tree Reduction inside Workgroup Shared Memory
        for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
            if (local_id.x < s) {
                sdata_sum[local_id.x] += sdata_sum[local_id.x + s];
                sdata_sq_sum[local_id.x] += sdata_sq_sum[local_id.x + s];
            }
            workgroupBarrier();
        }

        // Calculate Mean & InvStd
        if (local_id.x == 0u) {
            let n = f32(metaMatrix.reduce_size);
            let mu = sdata_sum[0] / n;
            let mean_sq = sdata_sq_sum[0] / n;
            let variance = max(0.0, mean_sq - (mu * mu));

            var inv_s: f32 = 0.0;
            if (variance > 0.0) {
                inv_s = inverseSqrt(variance + metaMatrix.eps);
            }

            saved_mean[out_idx] = mu;
            inv_std[out_idx] = inv_s;

            sdata_sum[0] = mu;
            sdata_sq_sum[0] = inv_s;
        }
        workgroupBarrier();

        let mu = sdata_sum[0];
        let inv_s = sdata_sq_sum[0];

        // =========================================================================
        // PASS 2: Scale and Shift Output Y
        // =========================================================================
        r = local_id.x;
        while (r < metaMatrix.reduce_size) {
            let physical_idx = base_offset + r * metaMatrix.inner_size;
            let x_val = X[physical_idx];

            Y[physical_idx] = (x_val - mu) * inv_s * gamma[r] + beta[r];
            r += WORKGROUP_SIZE;
        }

        i += inner_stride;
    }
}