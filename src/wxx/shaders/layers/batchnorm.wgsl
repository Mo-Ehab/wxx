struct BatchNormMeta {
    batch: u32,
    channels: u32,
    height: u32,
    width: u32,
    eps: f32,
    momentum: f32, // e.g. 0.1 for running stats EMA
    is_training: u32, // 1 = Training, 0 = Eval
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> X: array<f32>;             // (B, C, H, W)
@group(0) @binding(1) var<storage, read> gamma: array<f32>;         // (C,)
@group(0) @binding(2) var<storage, read> beta: array<f32>;          // (C,)
@group(0) @binding(3) var<storage, read_write> running_mean: array<f32>; // (C,)
@group(0) @binding(4) var<storage, read_write> running_var: array<f32>;  // (C,)
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;       // (B, C, H, W)
@group(0) @binding(6) var<storage, read_write> saved_mean: array<f32>; // (C,) saved for backward
@group(0) @binding(7) var<storage, read_write> inv_std: array<f32>;    // (C,) saved for backward
@group(0) @binding(8) var<uniform> metamatrix: BatchNormMeta;

const WORKGROUP_SIZE: u32 = 256u;
var<workgroup> sdata_sum: array<f32, 256>;
var<workgroup> sdata_sq: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let c = workgroup_id.x; // Each workgroup processes 1 channel c
    if (c >= metamatrix.channels) { return; }

    let tid = local_id.x;
    let hw = metamatrix.height * metamatrix.width;
    let total_m = metamatrix.batch * hw; // Total elements per channel M = B * H * W
    let plane_stride = metamatrix.channels * hw;

    if (metamatrix.is_training == 1u) {
        // =========================================================================
        // TRAINING MODE: Calculate Batch Mean & Variance across (B, H, W)
        // =========================================================================
        var thread_sum: f32 = 0.0;
        var thread_sq: f32 = 0.0;

        var i = tid;
        while (i < total_m) {
            let b = i / hw;
            let spatial = i % hw;
            let physical_idx = b * plane_stride + c * hw + spatial;

            let x_val = X[physical_idx];
            thread_sum += x_val;
            thread_sq += x_val * x_val;

            i += WORKGROUP_SIZE;
        }

        sdata_sum[tid] = thread_sum;
        sdata_sq[tid] = thread_sq;
        workgroupBarrier();

        // Parallel Tree Reduction
        for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
            if (tid < s) {
                sdata_sum[tid] += sdata_sum[tid + s];
                sdata_sq[tid] += sdata_sq[tid + s];
            }
            workgroupBarrier();
        }

        if (tid == 0u) {
            let m_f32 = f32(total_m);
            let mu = sdata_sum[0] / m_f32;
            let mean_sq = sdata_sq[0] / m_f32;
            let var_b = max(0.0, mean_sq - (mu * mu));

            let inv_s = inverseSqrt(var_b + metamatrix.eps);

            saved_mean[c] = mu;
            inv_std[c] = inv_s;

            // Exponential Moving Average update for inference running stats
            let mom = metamatrix.momentum;
            running_mean[c] = (1.0 - mom) * running_mean[c] + mom * mu;

            // Unbiased sample variance for running variance estimation
            let unbiased_var = select(var_b, var_b * (m_f32 / (m_f32 - 1.0)), total_m > 1u);
            running_var[c] = (1.0 - mom) * running_var[c] + mom * unbiased_var;

            sdata_sum[0] = mu;
            sdata_sq[0] = inv_s;
        }
        workgroupBarrier();

        let mu = sdata_sum[0];
        let inv_s = sdata_sq[0];

        // Normalize Y
        let g = gamma[c];
        let b_val = beta[c];

        var idx = tid;
        while (idx < total_m) {
            let b = idx / hw;
            let spatial = idx % hw;
            let physical_idx = b * plane_stride + c * hw + spatial;

            Y[physical_idx] = (X[physical_idx] - mu) * inv_s * g + b_val;
            idx += WORKGROUP_SIZE;
        }
    } else {
        // =========================================================================
        // EVALUATION MODE: Use frozen running_mean & running_var
        // =========================================================================
        let r_mean = running_mean[c];
        let r_var = running_var[c];
        let inv_s = inverseSqrt(r_var + metamatrix.eps);
        let g = gamma[c];
        let b_val = beta[c];

        var idx = tid;
        while (idx < total_m) {
            let b = idx / hw;
            let spatial = idx % hw;
            let physical_idx = b * plane_stride + c * hw + spatial;

            Y[physical_idx] = (X[physical_idx] - r_mean) * inv_s * g + b_val;
            idx += WORKGROUP_SIZE;
        }
    }
}