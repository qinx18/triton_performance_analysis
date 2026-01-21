import triton
import triton.language as tl

@triton.jit
def s233_kernel(aa, bb, cc, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    i_start = pid * BLOCK_SIZE
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = i_start + i_offsets
    i_mask = i_indices < N
    
    # Filter valid i indices (starting from 1)
    valid_i_mask = i_mask & (i_indices >= 1)
    
    # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
    for j in range(1, N):
        aa_prev_ptrs = aa + (j - 1) * N + i_indices
        aa_curr_ptrs = aa + j * N + i_indices
        cc_ptrs = cc + j * N + i_indices
        
        aa_prev = tl.load(aa_prev_ptrs, mask=valid_i_mask, other=0.0)
        cc_val = tl.load(cc_ptrs, mask=valid_i_mask, other=0.0)
        
        result = aa_prev + cc_val
        tl.store(aa_curr_ptrs, result, mask=valid_i_mask)
    
    # Second loop: bb[j][i] = bb[j][i-1] + cc[j][i]
    for j in range(1, N):
        bb_curr_ptrs = bb + j * N + i_indices
        bb_prev_ptrs = bb + j * N + (i_indices - 1)
        cc_ptrs = cc + j * N + i_indices
        
        bb_prev = tl.load(bb_prev_ptrs, mask=valid_i_mask, other=0.0)
        cc_val = tl.load(cc_ptrs, mask=valid_i_mask, other=0.0)
        
        result = bb_prev + cc_val
        tl.store(bb_curr_ptrs, result, mask=valid_i_mask)

def s233_triton(aa, bb, cc):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s233_kernel[grid](
        aa, bb, cc, N, BLOCK_SIZE
    )