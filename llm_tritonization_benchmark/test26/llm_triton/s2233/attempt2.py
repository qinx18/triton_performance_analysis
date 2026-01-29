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
    
    # Sequential loop over i dimension
    for i in range(1, N):
        # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
        for j in range(1, N):
            aa_curr_ptr = aa_ptr + j * N + i
            aa_prev_ptr = aa_ptr + (j - 1) * N + i
            cc_ptr_curr = cc_ptr + j * N + i
            
            aa_prev_val = tl.load(aa_prev_ptr)
            cc_val = tl.load(cc_ptr_curr)
            aa_new_val = aa_prev_val + cc_val
            tl.store(aa_curr_ptr, aa_new_val)
        
        # Second loop: bb[i][j] = bb[i-1][j] + cc[i][j]
        if i >= 1:
            for j in range(1, N):
                bb_curr_ptr = bb_ptr + i * N + j
                bb_prev_ptr = bb_ptr + (i - 1) * N + j
                cc_ptr_curr = cc_ptr + i * N + j
                
                bb_prev_val = tl.load(bb_prev_ptr)
                cc_val = tl.load(cc_ptr_curr)
                bb_new_val = bb_prev_val + cc_val
                tl.store(bb_curr_ptr, bb_new_val)

def s2233_triton(aa, bb, cc):
    N = aa.shape[0]
    
    grid = (1,)
    s2233_kernel[grid](
        aa, bb, cc,
        N, 1
    )