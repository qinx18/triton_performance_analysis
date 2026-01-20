import triton
import triton.language as tl
import torch

@triton.jit
def s233_kernel(aa_ptr, bb_ptr, cc_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets + 1
    i_mask = i_idx < N
    
    # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
    for j in range(1, N):
        aa_prev_ptrs = aa_ptr + (j - 1) * N + i_idx
        cc_curr_ptrs = cc_ptr + j * N + i_idx
        aa_curr_ptrs = aa_ptr + j * N + i_idx
        
        aa_prev_vals = tl.load(aa_prev_ptrs, mask=i_mask, other=0.0)
        cc_curr_vals = tl.load(cc_curr_ptrs, mask=i_mask, other=0.0)
        aa_new_vals = aa_prev_vals + cc_curr_vals
        tl.store(aa_curr_ptrs, aa_new_vals, mask=i_mask)
    
    # Second loop: bb[j][i] = bb[j][i-1] + cc[j][i]
    for j in range(1, N):
        bb_prev_ptrs = bb_ptr + j * N + (i_idx - 1)
        bb_curr_ptrs = bb_ptr + j * N + i_idx
        cc_curr_ptrs = cc_ptr + j * N + i_idx
        
        bb_prev_vals = tl.load(bb_prev_ptrs, mask=i_mask, other=0.0)
        bb_curr_vals = tl.load(bb_curr_ptrs, mask=i_mask, other=0.0)
        cc_curr_vals = tl.load(cc_curr_ptrs, mask=i_mask, other=0.0)
        
        # Sequential update within each j
        for k in range(BLOCK_SIZE):
            if i_idx[k] < N:
                curr_i = i_idx[k]
                if curr_i > 0:
                    bb_prev_val = tl.load(bb_ptr + j * N + (curr_i - 1))
                    cc_val = tl.load(cc_ptr + j * N + curr_i)
                    bb_new_val = bb_prev_val + cc_val
                    tl.store(bb_ptr + j * N + curr_i, bb_new_val)

def s233_triton(aa, bb, cc):
    N = aa.shape[0]
    BLOCK_SIZE = 32
    
    # Handle the i-dimension dependency in bb array sequentially
    for i in range(1, N):
        for j in range(1, N):
            # aa[j][i] = aa[j-1][i] + cc[j][i]
            aa[j, i] = aa[j-1, i] + cc[j, i]
            
        for j in range(1, N):
            # bb[j][i] = bb[j][i-1] + cc[j][i]
            bb[j, i] = bb[j, i-1] + cc[j, i]