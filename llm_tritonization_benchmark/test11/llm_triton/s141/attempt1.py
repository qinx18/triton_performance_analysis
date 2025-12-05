import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(
    bb_ptr,
    flat_2d_array_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Calculate initial k value
    k = (i + 1) * i // 2 + i
    
    # Process elements from j=i to LEN_2D-1
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_start in range(i, LEN_2D, BLOCK_SIZE):
        current_j = j_start + j_offsets
        j_mask = (current_j < LEN_2D) & (current_j >= i)
        
        # Calculate current k values for this block
        k_values = k + tl.cumsum(current_j, axis=0) + (current_j - i)
        
        # Load bb[j][i] values
        bb_indices = current_j * LEN_2D + i
        bb_mask = j_mask & (bb_indices < LEN_2D * LEN_2D)
        bb_vals = tl.load(bb_ptr + bb_indices, mask=bb_mask, other=0.0)
        
        # Load current flat_2d_array values
        flat_mask = j_mask & (k_values < LEN_2D * LEN_2D)
        current_vals = tl.load(flat_2d_array_ptr + k_values, mask=flat_mask, other=0.0)
        
        # Update flat_2d_array
        new_vals = current_vals + bb_vals
        tl.store(flat_2d_array_ptr + k_values, new_vals, mask=flat_mask)
        
        # Update k for next iteration
        if j_start + BLOCK_SIZE < LEN_2D:
            last_valid_j = min(j_start + BLOCK_SIZE - 1, LEN_2D - 1)
            k += tl.sum(tl.arange(j_start, last_valid_j + 1) + 1)

def s141_triton(bb, flat_2d_array):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 64
    
    grid = (LEN_2D,)
    
    s141_kernel[grid](
        bb,
        flat_2d_array,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )