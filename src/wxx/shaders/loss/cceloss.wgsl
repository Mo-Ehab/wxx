struct Meta {
    rows: u32,  // Batch size B
    cols: u32,  // Number of categories K
};

@group(0) @binding(0) var<storage, read_write> Z: array<f32>;        // Logits (B, K)
@group(0) @binding(1) var<storage, read_write> Targets: array<u32>;  // Category targets (B)
@group(0) @binding(2) var<storage, read_write> Loss: array<f32>;     // Output Loss (B)
@group(0) @binding(3) var<storage, read_write> dZ: array<f32>;       // Output Gradients (B, K)
@group(0) @binding(4) var<uniform> metamatrix: Meta;

var<workgroup> s_max: array<f32, 256>;
var<workgroup> s_sum: array<f32, 256>;

// Standard numerically-stable softmax cross-entropy:
//   loss = -z[target] + log(sum_j exp(z[j]))
//   grad[j] = softmax(z)[j] - 1{j == target}
//
// Shift all logits by max(z) before exp to avoid overflow.

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let row = wg_id.x;
    if (row >= metamatrix.rows) { return; }

    let tid        = local_id.x;
    let K          = metamatrix.cols;
    let row_offset = row * K;
    let target_idx = Targets[row];

    // ------------------------------------------------------------------
    // Pass 1 - find max logit for numerical stability
    // ------------------------------------------------------------------
    var local_max: f32 = -1e38;
    var c = tid;
    while (c < K) {
        local_max = max(local_max, Z[row_offset + c]);
        c += 256u;
    }
    s_max[tid] = local_max;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) { s_max[tid] = max(s_max[tid], s_max[tid + s]); }
        workgroupBarrier();
    }
    let z_max = s_max[0];

    // ------------------------------------------------------------------
    // Pass 2 - sum exp(z - z_max)
    // ------------------------------------------------------------------
    var local_sum: f32 = 0.0;
    c = tid;
    while (c < K) {
        local_sum += exp(Z[row_offset + c] - z_max);
        c += 256u;
    }
    s_sum[tid] = local_sum;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) { s_sum[tid] += s_sum[tid + s]; }
        workgroupBarrier();
    }
    let sum_exp     = s_sum[0];
    let log_sum_exp = log(sum_exp) + z_max;   // log sum_j exp(z[j])

    // loss = log_sum_exp - z[target]
    if (tid == 0u) {
        Loss[row] = log_sum_exp - Z[row_offset + target_idx];
    }

    // ------------------------------------------------------------------
    // Pass 3 - write gradients: softmax[j] - 1{j == target}
    // ------------------------------------------------------------------
    c = tid;
    while (c < K) {
        let sm_j      = exp(Z[row_offset + c] - z_max) / sum_exp;
        let is_target = select(0.0, 1.0, c == target_idx);
        dZ[row_offset + c] = sm_j - is_target;
        c += 256u;
    }
}
