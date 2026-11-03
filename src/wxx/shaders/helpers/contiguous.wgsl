struct ContiguousMeta {
    numel: u32,               // Total elements in tensor
    rank: u32,                // Number of active dimensions (1 to 6)
    offset: u32,              // Input tensor starting offset in VRAM
    _pad0: u32,               // 16-byte alignment padding
    shape: array<u32, 6>,     // Logical shape dimensions
    strides: array<u32, 6>,   // Strided offsets of the input view
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,               // Total struct size: 80 bytes (20 uint32s)
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> metamatrix: ContiguousMeta;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total_elements = metamatrix.numel;
    let grid_stride = num_workgroups.x * WORKGROUP_SIZE;

    var linear_idx = global_id.x;

    // Grid-stride loop over output elements
    while (linear_idx < total_elements) {
        if (linear_idx >= arrayLength(&dst)) { break; }
        var temp_idx = linear_idx;
        var strided_src_offset = metamatrix.offset;

        // Decompose 1D linear index into coordinates (right-to-left)
        // and compute non-contiguous VRAM source offset
        for (var i = 0u; i < metamatrix.rank; i = i + 1u) {
            let dim_idx = metamatrix.rank - 1u - i; // Process last dimension first
            let dim_size = metamatrix.shape[dim_idx];

            let coord = temp_idx % dim_size;
            temp_idx = temp_idx / dim_size;

            strided_src_offset += coord * metamatrix.strides[dim_idx];
        }

        // Read strided non-contiguous memory -> Write flat contiguous memory
        dst[linear_idx] = src[strided_src_offset];

        linear_idx += grid_stride;
    }
}