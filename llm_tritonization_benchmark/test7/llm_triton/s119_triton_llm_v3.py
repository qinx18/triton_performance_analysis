import torch
import triton
import triton.language as tl

@triton.jit
def s119_kernel(aa, bb, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Sequential processing over i (from 1 to LEN_2D-1)
    for i in range(1, LEN_2D):
        # Vectorized processing over j dimension
        pid = tl.program_id(0)
        j_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        j_mask = (j_offsets < LEN_2D) & (j_offsets >= 1)
        
        # Calculate linear indices for 2D array access
        curr_indices = i * LEN_2D + j_offsets
        prev_indices = (i - 1) * LEN_2D + (j_offsets - 1)
        
        # Load values
        aa_prev = tl.load(aa + prev_indices, mask=j_mask, other=0.0)
        bb_curr = tl.load(bb + curr_indices, mask=j_mask, other=0.0)
        
        # Compute and store result
        result = aa_prev + bb_curr
        tl.store(aa + curr_indices, result, mask=j_mask)

def s119_triton(aa, bb):
    LEN_2D = int(aa.shape[0])
    BLOCK_SIZE = 64
    
    # Launch kernel with grid size covering j dimension
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    s119_kernel[grid](aa, bb, LEN_2D, BLOCK_SIZE)