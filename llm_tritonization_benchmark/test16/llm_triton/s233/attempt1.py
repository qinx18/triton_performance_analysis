import torch
import triton
import triton.language as tl

@triton.jit
def s233_kernel(aa_ptr, bb_ptr, cc_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets + 1
    i_mask = i_idx < N
    
    for j in range(1, N):
        # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
        aa_prev_ptr = aa_ptr + (j - 1) * N + i_idx
        aa_curr_ptr = aa_ptr + j * N + i_idx
        cc_ptr_curr = cc_ptr + j * N + i_idx
        
        aa_prev = tl.load(aa_prev_ptr, mask=i_mask, other=0.0)
        cc_curr = tl.load(cc_ptr_curr, mask=i_mask, other=0.0)
        aa_new = aa_prev + cc_curr
        tl.store(aa_curr_ptr, aa_new, mask=i_mask)
        
        # Second loop: bb[j][i] = bb[j][i-1] + cc[j][i]
        bb_prev_ptr = bb_ptr + j * N + (i_idx - 1)
        bb_curr_ptr = bb_ptr + j * N + i_idx
        
        bb_prev = tl.load(bb_prev_ptr, mask=i_mask, other=0.0)
        bb_new = bb_prev + cc_curr
        tl.store(bb_curr_ptr, bb_new, mask=i_mask)

def s233_triton(aa, bb, cc):
    N = aa.shape[0]
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(N - 1, BLOCK_SIZE),)
    
    s233_kernel[grid](
        aa, bb, cc,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )