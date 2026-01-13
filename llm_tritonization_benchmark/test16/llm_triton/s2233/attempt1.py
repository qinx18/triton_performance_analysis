import torch
import triton
import triton.language as tl

@triton.jit
def s2233_kernel(aa_ptr, bb_ptr, cc_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets + 1
    i_mask = i_idx < N
    
    for j in range(1, N):
        # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
        aa_prev_ptr = aa_ptr + (j-1) * N
        aa_curr_ptr = aa_ptr + j * N
        cc_curr_ptr = cc_ptr + j * N
        
        aa_prev = tl.load(aa_prev_ptr + i_idx, mask=i_mask)
        cc_val = tl.load(cc_curr_ptr + i_idx, mask=i_mask)
        aa_new = aa_prev + cc_val
        tl.store(aa_curr_ptr + i_idx, aa_new, mask=i_mask)
        
        # Second loop: bb[i][j] = bb[i-1][j] + cc[i][j]
        bb_prev_ptr = bb_ptr + (i_idx - 1) * N + j
        bb_curr_ptr = bb_ptr + i_idx * N + j
        cc_ij_ptr = cc_ptr + i_idx * N + j
        
        bb_prev = tl.load(bb_prev_ptr, mask=i_mask)
        cc_ij = tl.load(cc_ij_ptr, mask=i_mask)
        bb_new = bb_prev + cc_ij
        tl.store(bb_curr_ptr, bb_new, mask=i_mask)

def s2233_triton(aa, bb, cc):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(N - 1, BLOCK_SIZE),)
    
    s2233_kernel[grid](
        aa, bb, cc, N, BLOCK_SIZE
    )