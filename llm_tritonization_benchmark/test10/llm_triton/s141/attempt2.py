import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(
    flat_2d_array_ptr,
    bb_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Calculate initial k
    k = i * (i + 1) // 2 + i
    
    # Process elements in blocks
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_start in range(i, LEN_2D, BLOCK_SIZE):
        current_j = j_start + j_offsets
        mask = (current_j < LEN_2D) & (current_j >= i)
        
        # Load bb[j][i] values
        bb_indices = current_j * LEN_2D + i
        bb_vals = tl.load(bb_ptr + bb_indices, mask=mask, other=0.0)
        
        # Calculate k indices for this block
        k_indices = k + (current_j - i) * (current_j + i + 1) // 2
        
        # Load current values from flat_2d_array
        current_vals = tl.load(flat_2d_array_ptr + k_indices, mask=mask, other=0.0)
        
        # Add bb values
        new_vals = current_vals + bb_vals
        
        # Store back
        tl.store(flat_2d_array_ptr + k_indices, new_vals, mask=mask)

def s141_triton(bb, flat_2d_array):
    LEN_2D = bb.shape[0]
    
    BLOCK_SIZE = 64
    grid = (LEN_2D,)
    
    s141_kernel[grid](
        flat_2d_array,
        bb,
        LEN_2D,
        BLOCK_SIZE,
    )