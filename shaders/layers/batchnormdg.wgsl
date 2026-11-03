struct BatchNormMeta {
    batch: u32,
    channels: u32,
    height: u32,
    width: u32,
    eps: f32,
    momentum: f32,
    is_training: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> dY: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> saved_mean: array<f32>;
@group(0) @binding(3) var<storage, read> inv_std: array<f32>;
@group(0) @binding(4) var<storage, read_write> dgamma: array<f32>;
@group(0) @binding(5) var<storage, read_write> dbeta: array<f32>;
@group(0) @binding(6) var<uniform> metamatrix: BatchNormMeta;

const WORKGROUP_SIZE: u32 = 256u;
var<workgroup> sdata_dgamma: array<f32, 256>;
var<workgroup> sdata_dbeta: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let c = workgroup_id.x;
    if (c >= metamatrix.channels) { return; }

    let tid = local_id.x;
    let hw = metamatrix.height * metamatrix.width;
    let total_m = metamatrix.batch * hw;
    let plane_stride = metamatrix.channels * hw;

    let mu = saved_mean[c];
    let inv_s = inv_std[c];

    var thread_dgamma: f32 = 0.0;
    var thread_dbeta: f32 = 0.0;

    var i = tid;
    while (i < total_m) {
        let b = i / hw;
        let spatial = i % hw;
        let physical_idx = b * plane_stride + c * hw + spatial;

        let dy_val = dY[physical_idx];
        let x_hat = (X[physical_idx] - mu) * inv_s;

        thread_dgamma += dy_val * x_hat;
        thread_dbeta += dy_val;

        i += WORKGROUP_SIZE;
    }

    sdata_dgamma[tid] = thread_dgamma;
    sdata_dbeta[tid] = thread_dbeta;
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
        dgamma[c] = sdata_dgamma[0];
        dbeta[c] = sdata_dbeta[0];
    }
}