import triton
import triton.language as tl

@triton.jit
def s233_kernel(aa_ptr, bb_ptr, cc_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = pid * BLOCK_SIZE + i_offsets + 1
    i_mask = i_indices < N
    
    for j in range(1, N):
        # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
        aa_prev_ptrs = aa_ptr + (j-1) * N + i_indices
        aa_curr_ptrs = aa_ptr + j * N + i_indices
        cc_ptrs = cc_ptr + j * N + i_indices
        
        aa_prev = tl.load(aa_prev_ptrs, mask=i_mask, other=0.0)
        cc_vals = tl.load(cc_ptrs, mask=i_mask, other=0.0)
        aa_new = aa_prev + cc_vals
        tl.store(aa_curr_ptrs, aa_new, mask=i_mask)

@triton.jit
def s233_bb_kernel(bb_ptr, cc_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_indices = pid * BLOCK_SIZE + j_offsets + 1
    j_mask = j_indices < N
    
    for i in range(1, N):
        # Second loop: bb[j][i] = bb[j][i-1] + cc[j][i]
        bb_prev_ptrs = bb_ptr + j_indices * N + (i-1)
        bb_curr_ptrs = bb_ptr + j_indices * N + i
        cc_ptrs = cc_ptr + j_indices * N + i
        
        bb_prev = tl.load(bb_prev_ptrs, mask=j_mask, other=0.0)
        cc_vals = tl.load(cc_ptrs, mask=j_mask, other=0.0)
        bb_new = bb_prev + cc_vals
        tl.store(bb_curr_ptrs, bb_new, mask=j_mask)

def s233_triton(aa, bb, cc):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(N - 1, BLOCK_SIZE),)
    s233_kernel[grid](aa, bb, cc, N, BLOCK_SIZE)
    s233_bb_kernel[grid](bb, cc, N, BLOCK_SIZE)