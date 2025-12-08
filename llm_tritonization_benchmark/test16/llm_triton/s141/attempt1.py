import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for i dimension
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Calculate starting k for this i
    k = (i + 1) * (i + 1 - 1) // 2 + (i + 1) - 1
    
    # Process j loop in blocks
    j_start = i
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_block in range(j_start, LEN_2D, BLOCK_SIZE):
        # Current j indices
        j_indices = j_block + j_offsets
        j_mask = (j_indices < LEN_2D) & (j_indices >= i)
        
        if not tl.any(j_mask):
            break
            
        # Calculate k indices for this block
        k_indices = k + tl.cumsum(j_indices + 1, axis=0) - (i + 1)
        
        # Load bb values: bb[j][i]
        bb_offsets = j_indices * LEN_2D + i
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
        
        # Load current flat_2d_array values
        flat_vals = tl.load(flat_2d_array_ptr + k_indices, mask=j_mask, other=0.0)
        
        # Update flat_2d_array
        updated_vals = flat_vals + bb_vals
        tl.store(flat_2d_array_ptr + k_indices, updated_vals, mask=j_mask)
        
        # Update k for next block
        valid_j_indices = tl.where(j_mask, j_indices, 0)
        last_valid_j = tl.max(valid_j_indices)
        k += tl.sum(tl.where(j_mask, j_indices + 1, 0))

def s141_triton(bb, flat_2d_array):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 64
    
    grid = (LEN_2D,)
    
    s141_kernel[grid](
        bb, flat_2d_array,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )