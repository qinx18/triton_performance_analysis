import triton
import triton.language as tl
import torch

@triton.jit
def s2233_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0) + 1
    
    if i >= LEN_2D:
        return
    
    # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
    # This has a dependency on j dimension, so we serialize it
    for j in range(1, LEN_2D):
        aa_prev_idx = (j - 1) * LEN_2D + i
        aa_curr_idx = j * LEN_2D + i
        cc_idx = j * LEN_2D + i
        
        aa_prev_val = tl.load(aa_ptr + aa_prev_idx)
        cc_val = tl.load(cc_ptr + cc_idx)
        result = aa_prev_val + cc_val
        tl.store(aa_ptr + aa_curr_idx, result)
    
    # Second loop: bb[i][j] = bb[i-1][j] + cc[i][j]  
    # This has a dependency on i dimension, but we're processing one i at a time
    # We can vectorize the j dimension
    j_offsets = tl.arange(0, BLOCK_SIZE) + 1
    
    for j_start in range(1, LEN_2D, BLOCK_SIZE):
        j_indices = j_start + tl.arange(0, BLOCK_SIZE)
        mask = j_indices < LEN_2D
        
        bb_prev_indices = (i - 1) * LEN_2D + j_indices
        bb_curr_indices = i * LEN_2D + j_indices
        cc_indices = i * LEN_2D + j_indices
        
        bb_prev_vals = tl.load(bb_ptr + bb_prev_indices, mask=mask)
        cc_vals = tl.load(cc_ptr + cc_indices, mask=mask)
        results = bb_prev_vals + cc_vals
        tl.store(bb_ptr + bb_curr_indices, results, mask=mask)

def s2233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Launch kernel with one thread per i index (excluding i=0)
    grid = (LEN_2D - 1,)
    
    s2233_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )