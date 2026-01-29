import triton
import triton.language as tl

@triton.jit
def s2233_kernel(aa_ptr, bb_ptr, cc_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets + 1
    i_mask = i_idx < N
    
    # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
    for j in range(1, N):
        aa_prev_ptrs = aa_ptr + (j - 1) * N + i_idx
        aa_curr_ptrs = aa_ptr + j * N + i_idx
        cc_ptrs = cc_ptr + j * N + i_idx
        
        aa_prev_vals = tl.load(aa_prev_ptrs, mask=i_mask, other=0.0)
        cc_vals = tl.load(cc_ptrs, mask=i_mask, other=0.0)
        aa_new_vals = aa_prev_vals + cc_vals
        tl.store(aa_curr_ptrs, aa_new_vals, mask=i_mask)
    
    # Second loop: bb[i][j] = bb[i-1][j] + cc[i][j]
    for j in range(1, N):
        bb_prev_ptrs = bb_ptr + (i_idx - 1) * N + j
        bb_curr_ptrs = bb_ptr + i_idx * N + j
        cc_ptrs = cc_ptr + i_idx * N + j
        
        bb_prev_vals = tl.load(bb_prev_ptrs, mask=i_mask, other=0.0)
        cc_vals = tl.load(cc_ptrs, mask=i_mask, other=0.0)
        bb_new_vals = bb_prev_vals + cc_vals
        tl.store(bb_curr_ptrs, bb_new_vals, mask=i_mask)

def s2233_triton(aa, bb, cc):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    grid = (triton.cdiv(N - 1, BLOCK_SIZE),)
    
    s2233_kernel[grid](aa, bb, cc, N, BLOCK_SIZE)