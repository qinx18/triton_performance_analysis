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
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_block in range(i, LEN_2D, BLOCK_SIZE):
        # Current j indices
        j_indices = j_block + j_offsets
        j_mask = (j_indices < LEN_2D) & (j_indices >= i)
        
        # Calculate k indices for this block
        # k for j = k_start + sum(j' + 1 for j' in range(i, j))
        k_offset = tl.sum(tl.where(j_indices < j_block, tl.where(j_indices >= i, j_indices + 1, 0), 0))
        k_indices = k + k_offset + tl.cumsum(tl.where(j_mask, j_indices + 1, 0), axis=0) - tl.where(j_mask, j_indices + 1, 0)
        
        # Load bb values: bb[j][i]
        bb_offsets = j_indices * LEN_2D + i
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
        
        # Load current flat_2d_array values
        flat_vals = tl.load(flat_2d_array_ptr + k_indices, mask=j_mask, other=0.0)
        
        # Update flat_2d_array
        updated_vals = flat_vals + bb_vals
        tl.store(flat_2d_array_ptr + k_indices, updated_vals, mask=j_mask)

def s141_triton(bb, flat_2d_array):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 64
    
    grid = (LEN_2D,)
    
    s141_kernel[grid](
        bb, flat_2d_array,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )