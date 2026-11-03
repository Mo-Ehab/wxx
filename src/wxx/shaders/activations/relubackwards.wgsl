@group(0) @binding(0) var<storage, read> dataA : array<f32>;
@group(0) @binding(1) var<storage, read_write> dataOut : array<f32>;

const WORKGROUP_SIZE : u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id : vec3<u32>,
    @builtin(num_workgroups) num_workgroups : vec3<u32>
) {
    let total_elements = arrayLength(&dataOut);
    let stride = num_workgroups.x * WORKGROUP_SIZE;
    
    var i = global_id.x;
    while (i < total_elements) {
        if (i >= arrayLength(&dataOut)) { break; }
        dataOut[i] = select(0.0, 1.0, dataA[i] > 0.0);
        i += stride;
    }
}