struct RMSNorm3DMeta {
    outer_size: u32,
    reduce_size: u32,
    inner_size: u32,
    eps: f32,
};

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;         // Shape: (reduce_size,)
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<storage, read_write> inv_rms: array<f32>; // Shape: (outer_size * inner_size,)
@group(0) @binding(4) var<uniform> metaMatrix: RMSNorm3DMeta;

const WORKGROUP_SIZE: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

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

    // Grid-stride over spatial dimension i
    while (i < metaMatrix.inner_size) {
        let out_idx = o * metaMatrix.inner_size + i;

        // --- 1. Compute thread_sq_sum ---
        var thread_sq_sum: f32 = 0.0;
        var r = local_id.x;
        while (r < metaMatrix.reduce_size) {
            let physical_idx = o * (metaMatrix.reduce_size * metaMatrix.inner_size) + r * metaMatrix.inner_size + i;
            let x_val = X[physical_idx];
            thread_sq_sum += x_val * x_val;
            r += WORKGROUP_SIZE;
        }

        sdata[local_id.x] = thread_sq_sum;
        workgroupBarrier();

        // --- 2. Parallel Tree Reduction ---
        for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
            if (local_id.x < s) {
                sdata[local_id.x] += sdata[local_id.x + s];
            }
            workgroupBarrier();
        }

        // --- 3. Save inv_rms ---
        if (local_id.x == 0u) {
            let mean_sq = sdata[0] / f32(metaMatrix.reduce_size);
            var r_val: f32 = 0.0;
            if (mean_sq > 0.0) {
                r_val = inverseSqrt(mean_sq + metaMatrix.eps);
            }
            inv_rms[out_idx] = r_val;
            sdata[0] = r_val;
        }
        workgroupBarrier();

        let r_val = sdata[0];

        // --- 4. Scale Y ---
        r = local_id.x;
        while (r < metaMatrix.reduce_size) {
            let physical_idx = o * (metaMatrix.reduce_size * metaMatrix.inner_size) + r * metaMatrix.inner_size + i;
            Y[physical_idx] = X[physical_idx] * r_val * gamma[r];
            r += WORKGROUP_SIZE;
        }

        // Advance to next spatial pixel block if inner_size > 65535
        i += inner_stride;
    }
}