import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(aa_ptr, bb_ptr, flat_2d_array_ptr, len_2d, k_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(len_2d):
        j_idx = pid * BLOCK_SIZE + j_offsets
        j_mask = j_idx < len_2d
        
        # Load bb values for current i
        bb_ptrs = bb_ptr + j_idx * len_2d + i
        bb_vals = tl.load(bb_ptrs, mask=j_mask, other=0.0)
        
        # Load aa values for current i  
        aa_ptrs = aa_ptr + j_idx * len_2d + i
        aa_vals = tl.load(aa_ptrs, mask=j_mask, other=0.0)
        
        # Check condition for each j in this block
        condition = bb_vals > 0.0
        
        # Process each element sequentially to maintain k ordering
        for j_local in range(BLOCK_SIZE):
            global_j = pid * BLOCK_SIZE + j_local
            if global_j < len_2d:
                if condition[j_local]:
                    # Atomically increment k and get the index
                    k_val = tl.atomic_add(k_ptr, 1)
                    # Store the value
                    tl.store(flat_2d_array_ptr + k_val, aa_vals[j_local])

def s343_triton(aa, bb, flat_2d_array):
    len_2d = aa.shape[0]
    
    # Create counter on GPU
    k_tensor = torch.tensor([-1], dtype=torch.int32, device=aa.device)
    
    BLOCK_SIZE = 64
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s343_kernel[grid](
        aa, bb, flat_2d_array, len_2d, k_tensor, BLOCK_SIZE
    )