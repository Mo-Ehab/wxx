struct SliceAssignMeta { 
    numel: u32,             // Total elements in the target slice
    rank: u32,              // Active rank of target slice
    is_scalar: u32,         // 1 = assigning scalar_val, 0 = reading src_buffer
    scalar_val: f32,
    dst_offset: u32,
    src_offset: u32,
    _pad0: u32,
    _pad1: u32,
    dst_shape: array<u32, 6>,
    dst_strides: array<u32, 6>,
    src_strides: array<u32, 6>,
};

@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> metamatrix: SliceAssignMeta;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total_elements = metamatrix.numel;
    let stride = num_workgroups.x * WORKGROUP_SIZE;

    var idx = global_id.x;
    while (idx < total_elements) {
        if (idx >= arrayLength(&dst)) { break; }
        var temp_idx = idx;
        var physical_dst_offset = metamatrix.dst_offset;
        var physical_src_offset = metamatrix.src_offset;

        // Decompose linear thread ID into multi-dimensional coordinates
        for (var i = 0u; i < metamatrix.rank; i = i + 1u) {
            let dim_idx = metamatrix.rank - 1u - i; // Right-to-left
            let dim_size = metamatrix.dst_shape[dim_idx];

            let coord = temp_idx % dim_size;
            temp_idx = temp_idx / dim_size;

            physical_dst_offset += coord * metamatrix.dst_strides[dim_idx];
            physical_src_offset += coord * metamatrix.src_strides[dim_idx];
        }

        if (metamatrix.is_scalar == 1u) {
            dst[physical_dst_offset] = metamatrix.scalar_val;
        } else {
            dst[physical_dst_offset] = src[physical_src_offset];
        }

        idx += stride;
    }
}