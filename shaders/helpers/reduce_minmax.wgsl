struct ReduceMeta {
    outer_size: u32,
    reduce_size: u32,
    inner_size: u32,
    op: u32,          // 0 = max, 1 = min
};

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_indices: array<u32>;
@group(0) @binding(3) var<uniform> metamatrix: ReduceMeta;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> sdata_val: array<f32, 256>;
var<workgroup> sdata_idx: array<u32, 256>;

fn is_better(candidate: f32, current: f32, op: u32) -> bool {
    if (op == 0u) {
        return candidate > current;
    }
    return candidate < current;
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let total_outputs = metamatrix.outer_size * metamatrix.inner_size;
    let out_idx = workgroup_id.x;

    if (out_idx >= total_outputs) {
        return;
    }

    let o = out_idx / metamatrix.inner_size;
    let i = out_idx % metamatrix.inner_size;

    // 1. Thread-local accumulation.
    var best_val: f32 = select(-1e38, 1e38, metamatrix.op == 1u);
    var best_arg: u32 = 0u;
    var r = local_id.x;

    while (r < metamatrix.reduce_size) {
        let in_idx = o * (metamatrix.reduce_size * metamatrix.inner_size)
            + r * metamatrix.inner_size + i;
        if (in_idx >= arrayLength(&input_data)) { break; }
        let val = input_data[in_idx];
        if (is_better(val, best_val, metamatrix.op)) {
            best_val = val;
            best_arg = r;
        }
        r += WORKGROUP_SIZE;
    }

    sdata_val[local_id.x] = best_val;
    sdata_idx[local_id.x] = best_arg;
    workgroupBarrier();

    // 2. Workgroup parallel tree reduction.
    for (var s = WORKGROUP_SIZE / 2u; s > 0u; s = s / 2u) {
        if (local_id.x < s) {
            let other_val = sdata_val[local_id.x + s];
            if (is_better(other_val, sdata_val[local_id.x], metamatrix.op)) {
                sdata_val[local_id.x] = other_val;
                sdata_idx[local_id.x] = sdata_idx[local_id.x + s];
            }
        }
        workgroupBarrier();
    }

    // 3. Thread 0 writes the final value and its reduction index.
    if (local_id.x == 0u) {
        output_values[out_idx] = sdata_val[0];
        output_indices[out_idx] = sdata_idx[0];
    }
}
