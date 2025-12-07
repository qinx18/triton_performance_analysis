import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Calculate initial k
    k = (i + 1) * (i + 1 - 1) // 2 + (i + 1) - 1
    
    # Process j from i to LEN_2D-1 in blocks
    for j_start in range(i, LEN_2D, BLOCK_SIZE):
        j_offsets = tl.arange(0, BLOCK_SIZE)
        j_indices = j_start + j_offsets
        j_mask = j_indices < LEN_2D
        
        # Load bb[j][i] values
        bb_offsets = j_indices * LEN_2D + i
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
        
        # Calculate k values for this block
        k_base = k + (j_start - i) * (j_start - i + 1) // 2
        if j_start == i:
            k_offsets = tl.arange(0, BLOCK_SIZE) * (tl.arange(0, BLOCK_SIZE) + 1) // 2
        else:
            # For subsequent blocks, k increments by (j_start - i) initial offset
            # plus cumulative increments
            k_offsets = tl.arange(0, BLOCK_SIZE) * (j_start + tl.arange(0, BLOCK_SIZE)) - \
                       tl.arange(0, BLOCK_SIZE) * (j_start - 1) // 2
        
        k_indices = k_base + k_offsets
        
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