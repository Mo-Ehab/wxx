struct ReduceMeta {
    outer_size: u32,
    reduce_size: u32,
    inner_size: u32,
    is_mean: u32, // 0 = sum, 1 = mean, 2 = sum of squares
};

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> metamat: ReduceMeta;

const WORKGROUP_SIZE: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total_outputs = metamat.outer_size * metamat.inner_size;
    let grid_stride_wg = num_workgroups.x;

    // Outer loop: Each workgroup steps through output indices
    var out_idx = workgroup_id.x;
    while (out_idx < total_outputs) {
        if (out_idx >= arrayLength(&output_data)) { break; }
        let o = out_idx / metamat.inner_size;
        let i = out_idx % metamat.inner_size;

        // 1. Thread-level grid-stride accumulation over 'Reduce' dimension
        var thread_sum: f32 = 0.0;
        var r = local_id.x;

        while (r < metamat.reduce_size) {
            let in_idx = o * (metamat.reduce_size * metamat.inner_size) + r * metamat.inner_size + i;
            if (in_idx >= arrayLength(&input_data)) { break; }
            let value = input_data[in_idx];
            if (metamat.is_mean == 2u) {
                // Fused sum-of-squares (used for gradient-norm clipping).
                thread_sum += value * value;
            } else {
                thread_sum += value;
            }
            r += WORKGROUP_SIZE;
        }

        sdata[local_id.x] = thread_sum;
        workgroupBarrier();

        // 2. Parallel Tree Reduction inside Workgroup
        for (var s = WORKGROUP_SIZE / 2u; s > 0u; s = s / 2u) {
            if (local_id.x < s) {
                sdata[local_id.x] += sdata[local_id.x + s];
            }
            workgroupBarrier();
        }

        // 3. Write output element
        if (local_id.x == 0u) {
            var final_val = sdata[0];
            if (metamat.is_mean == 1u) {
                final_val = final_val / f32(metamat.reduce_size);
            }
            output_data[out_idx] = final_val;
        }

        // Synchronize before reusing sdata for the next out_idx tile
        workgroupBarrier();

        out_idx += grid_stride_wg;
    }
}