import torch
import triton
import triton.language as tl

@triton.jit
def s232_kernel(aa, bb, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_indices = pid * BLOCK_SIZE + j_offsets + 1
    j_mask = j_indices < LEN_2D
    
    for i in range(1, LEN_2D):
        i_mask = i <= j_indices
        valid_mask = j_mask & i_mask
        
        # Current position indices
        curr_idx = j_indices * LEN_2D + i
        # Previous i position indices  
        prev_idx = j_indices * LEN_2D + (i - 1)
        
        # Load previous aa values and current bb values
        aa_prev = tl.load(aa + prev_idx, mask=valid_mask, other=0.0)
        bb_curr = tl.load(bb + curr_idx, mask=valid_mask, other=0.0)
        
        # Compute: aa[j][i] = aa[j][i-1] * aa[j][i-1] + bb[j][i]
        result = aa_prev * aa_prev + bb_curr
        
        # Store result
        tl.store(aa + curr_idx, result, mask=valid_mask)

def s232_triton(aa, bb):
    LEN_2D = aa.size(0)
    BLOCK_SIZE = 32
    
    # Number of j values to process (j goes from 1 to LEN_2D-1)
    j_size = LEN_2D - 1
    grid = (triton.cdiv(j_size, BLOCK_SIZE),)
    
    s232_kernel[grid](aa, bb, LEN_2D, BLOCK_SIZE)