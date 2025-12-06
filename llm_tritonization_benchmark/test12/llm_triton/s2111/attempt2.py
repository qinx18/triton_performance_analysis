import torch
import triton
import triton.language as tl

@triton.jit
def s2111_kernel(aa_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Sequential computation due to data dependencies
    for j in range(1, LEN_2D):
        for block_idx in range((LEN_2D - 1 + BLOCK_SIZE - 1) // BLOCK_SIZE):
            current_i_offsets = 1 + block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = current_i_offsets < LEN_2D
            
            # Load aa[j][i-1]
            left_offsets = j * LEN_2D + (current_i_offsets - 1)
            left_mask = mask & (current_i_offsets >= 1)
            left_vals = tl.load(aa_ptr + left_offsets, mask=left_mask, other=0.0)
            
            # Load aa[j-1][i]
            up_offsets = (j - 1) * LEN_2D + current_i_offsets
            up_vals = tl.load(aa_ptr + up_offsets, mask=mask, other=0.0)
            
            # Compute result
            result = (left_vals + up_vals) / 1.9
            
            # Store aa[j][i]
            store_offsets = j * LEN_2D + current_i_offsets
            tl.store(aa_ptr + store_offsets, result, mask=mask)

def s2111_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 32
    
    grid = (1,)  # Single program due to dependencies
    
    s2111_kernel[grid](
        aa, LEN_2D, BLOCK_SIZE
    )
    
    return aa