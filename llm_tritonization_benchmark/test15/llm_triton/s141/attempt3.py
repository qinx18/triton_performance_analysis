import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Calculate initial k: k = (i+1) * ((i+1) - 1) / 2 + (i+1) - 1
    # Simplifying: k = i * (i + 1) / 2 + i = i * (i + 3) / 2
    k = i * (i + 1) // 2 + i
    
    # Process j from i to LEN_2D-1 in blocks
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_start in range(i, LEN_2D, BLOCK_SIZE):
        j_indices = j_start + j_offsets
        j_mask = j_indices < LEN_2D
        
        # Load bb[j][i] values
        bb_offsets = j_indices * LEN_2D + i
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
        
        # Calculate k values for this block
        # k starts at initial value plus sum of (j+1) for j from i to j_start-1
        if j_start == i:
            k_start = k
        else:
            # Sum from i to j_start-1 of (j+1) = sum from (i+1) to j_start of j
            # = j_start*(j_start+1)/2 - i*(i+1)/2
            k_start = k + j_start * (j_start + 1) // 2 - i * (i + 1) // 2
        
        # For each element in block, k increments by (j+1)
        first_j = tl.maximum(j_start, i)
        k_increments = tl.cumsum(j_indices + 1, axis=0) - (first_j + 1)
        k_indices = k_start + k_increments
        
        # Load current flat_2d_array values
        flat_vals = tl.load(flat_2d_array_ptr + k_indices, mask=j_mask, other=0.0)
        
        # Add bb values
        new_vals = flat_vals + bb_vals
        
        # Store back
        tl.store(flat_2d_array_ptr + k_indices, new_vals, mask=j_mask)

def s141_triton(bb, flat_2d_array):
    LEN_2D = bb.shape[0]
    
    BLOCK_SIZE = 32
    grid = (LEN_2D,)
    
    s141_kernel[grid](
        bb, flat_2d_array, LEN_2D, BLOCK_SIZE
    )