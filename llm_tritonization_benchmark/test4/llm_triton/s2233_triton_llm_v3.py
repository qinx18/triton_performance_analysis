import torch
import triton
import triton.language as tl

@triton.jit
def s2233_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Process i dimension sequentially to maintain dependencies
    for i in range(1, LEN_2D):
        # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
        # Must be sequential in j dimension due to dependency on aa[j-1][i]
        for j in range(1, LEN_2D):
            aa_curr_idx = j * LEN_2D + i
            aa_prev_idx = (j - 1) * LEN_2D + i
            cc_idx = j * LEN_2D + i
            
            aa_prev_val = tl.load(aa_ptr + aa_prev_idx)
            cc_val = tl.load(cc_ptr + cc_idx)
            result = aa_prev_val + cc_val
            tl.store(aa_ptr + aa_curr_idx, result)
        
        # Second loop: bb[i][j] = bb[i-1][j] + cc[i][j]
        # Can be parallelized in j dimension
        j_offsets = tl.arange(0, BLOCK_SIZE)
        for j_start in range(1, LEN_2D, BLOCK_SIZE):
            j_indices = j_start + j_offsets
            mask = j_indices < LEN_2D
            
            bb_curr_indices = i * LEN_2D + j_indices
            bb_prev_indices = (i - 1) * LEN_2D + j_indices
            cc_indices = i * LEN_2D + j_indices
            
            bb_prev_vals = tl.load(bb_ptr + bb_prev_indices, mask=mask)
            cc_vals = tl.load(cc_ptr + cc_indices, mask=mask)
            results = bb_prev_vals + cc_vals
            tl.store(bb_ptr + bb_curr_indices, results, mask=mask)

def s2233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Launch single thread block since i dimension must be sequential
    grid = (1,)
    
    s2233_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )