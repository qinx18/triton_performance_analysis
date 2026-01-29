import triton
import triton.language as tl

@triton.jit
def s2233_kernel(
    aa_ptr, bb_ptr, cc_ptr,
    N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < N
    i_valid = (i_idx >= 1) & i_mask
    
    # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
    for j in range(1, N):
        aa_curr_ptrs = aa_ptr + j * N + i_idx
        aa_prev_ptrs = aa_ptr + (j - 1) * N + i_idx
        cc_ptrs = cc_ptr + j * N + i_idx
        
        aa_prev_vals = tl.load(aa_prev_ptrs, mask=i_valid)
        cc_vals = tl.load(cc_ptrs, mask=i_valid)
        aa_new_vals = aa_prev_vals + cc_vals
        tl.store(aa_curr_ptrs, aa_new_vals, mask=i_valid)
    
    # Second loop: bb[i][j] = bb[i-1][j] + cc[i][j]
    j_offsets = tl.arange(0, BLOCK_SIZE)
    for j_block_start in range(1, N, BLOCK_SIZE):
        j_idx = j_block_start + j_offsets
        j_mask = j_idx < N
        j_valid = (j_idx >= 1) & j_mask
        
        for i in range(1, N):
            bb_curr_ptrs = bb_ptr + i * N + j_idx
            bb_prev_ptrs = bb_ptr + (i - 1) * N + j_idx
            cc_ptrs = cc_ptr + i * N + j_idx
            
            bb_prev_vals = tl.load(bb_prev_ptrs, mask=j_valid)
            cc_vals = tl.load(cc_ptrs, mask=j_valid)
            bb_new_vals = bb_prev_vals + cc_vals
            tl.store(bb_curr_ptrs, bb_new_vals, mask=j_valid)

def s2233_triton(aa, bb, cc):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s2233_kernel[grid](
        aa, bb, cc,
        N, BLOCK_SIZE
    )