struct Norm3DMeta {
    outer_size: u32,
    reduce_size: u32,
    inner_size: u32,
    eps: f32,
};

@group(0) @binding(0) var<storage, read> dY: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> saved_mean: array<f32>;
@group(0) @binding(3) var<storage, read> inv_std: array<f32>;
@group(0) @binding(4) var<storage, read_write> dgamma: array<f32>;
@group(0) @binding(5) var<storage, read_write> dbeta: array<f32>;
@group(0) @binding(6) var<uniform> metaMatrix: Norm3DMeta;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> sdata_dgamma: array<f32, 256>;
var<workgroup> sdata_dbeta: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let r = workgroup_id.x; // Feature channel index
    if (r >= metaMatrix.reduce_size) { return; }

    let tid = local_id.x;
    var thread_sum_dgamma: f32 = 0.0;
    var thread_sum_dbeta: f32 = 0.0;

    let reduce_inner = metaMatrix.reduce_size * metaMatrix.inner_size;

    // Nested spatial/batch loops: 100% FREE OF INTEGER DIVISION AND MODULO!
    for (var o = 0u; o < metaMatrix.outer_size; o++) {
        let base_x = o * reduce_inner + r * metaMatrix.inner_size;
        let base_stat = o * metaMatrix.inner_size;

        var i = tid;
        // Coalesced Grid-Stride Loop along inner spatial elements
        while (i < metaMatrix.inner_size) {
            let physical_idx = base_x + i;
            let stat_idx = base_stat + i;

            let dy_val = dY[physical_idx];
            let x_val = X[physical_idx];
            let mu_val = saved_mean[stat_idx];
            let inv_s = inv_std[stat_idx];

            let x_hat = (x_val - mu_val) * inv_s;

            thread_sum_dgamma += dy_val * x_hat;
            thread_sum_dbeta += dy_val;

            i += WORKGROUP_SIZE;
        }
    }

    sdata_dgamma[tid] = thread_sum_dgamma;
    sdata_dbeta[tid] = thread_sum_dbeta;
    workgroupBarrier();

    // Parallel tree reduction
    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sdata_dgamma[tid] += sdata_dgamma[tid + s];
            sdata_dbeta[tid] += sdata_dbeta[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        dgamma[r] = sdata_dgamma[0];
        dbeta[r] = sdata_dbeta[0];
    }
}