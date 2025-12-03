import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(flat_2d_array_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
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
        
        # Calculate k offsets for this block
        k_offsets = tl.arange(0, BLOCK_SIZE)
        k_indices = k + k_offsets
        
        # Load bb[j][i] values
        bb_offsets = j_indices * LEN_2D + i
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
        
        # Load current flat_2d_array values
        flat_vals = tl.load(flat_2d_array_ptr + k_indices, mask=j_mask, other=0.0)
        
        # Add bb values to flat_2d_array
        result = flat_vals + bb_vals
        
        # Store back
        tl.store(flat_2d_array_ptr + k_indices, result, mask=j_mask)
        
        # Update k for next block
        if j_start + BLOCK_SIZE < LEN_2D:
            k += BLOCK_SIZE * (j_start + BLOCK_SIZE - i) + BLOCK_SIZE * (BLOCK_SIZE - 1) // 2

def s141_triton(bb, flat_2d_array):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 32
    
    grid = (LEN_2D,)
    
    s141_kernel[grid](
        flat_2d_array,
        bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )