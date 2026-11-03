struct Meta {
    num_tokens: u32,
    embed_dim: u32,
    vocab_size: u32,
    _pad0: u32
};

@group(0) @binding(0) var<storage, read> Indices: array<u32>;
@group(0) @binding(1) var<storage, read> GradOutput: array<f32>;
@group(0) @binding(2) var<storage, read_write> GradWeight: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> metamatrix: Meta;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total_elements = metamatrix.num_tokens * metamatrix.embed_dim;
    let stride = num_workgroups.x * WORKGROUP_SIZE;

    // Grid-stride loop: covers every element even when the dispatch is
    // clamped to the maximum number of workgroups per dimension.
    var thread_idx = global_id.x;
    while (thread_idx < total_elements) {
        let token_idx = thread_idx / metamatrix.embed_dim;
        let dim_idx = thread_idx % metamatrix.embed_dim;
        let vocab_id = Indices[token_idx];

        if (vocab_id < metamatrix.vocab_size) {
            let grad_val = GradOutput[thread_idx];
            let target_idx = vocab_id * metamatrix.embed_dim + dim_idx;

            // --- Inlined Atomic Float Addition (CAS Loop) ---
            var old_bits = atomicLoad(&GradWeight[target_idx]);
            loop {
                let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + grad_val);
                let res = atomicCompareExchangeWeak(&GradWeight[target_idx], old_bits, new_bits);
                if (res.exchanged) {
                    break;
                }
                old_bits = res.old_value;
            }
        }

        thread_idx += stride;
    }
}
