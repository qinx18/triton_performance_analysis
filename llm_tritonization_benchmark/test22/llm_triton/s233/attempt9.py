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
        cc_val_aa = tl.load(cc_ptrs, mask=valid_i_mask, other=0.0)
        
        result_aa = aa_prev + cc_val_aa
        tl.store(aa_curr_ptrs, result_aa, mask=valid_i_mask)
    
    # Second loop: bb[j][i] = bb[j][i-1] + cc[j][i]
    for j in range(1, N):
        for i_idx in range(1, N):
            i_match_mask = (i_indices == i_idx)
            final_mask = valid_i_mask & i_match_mask
            
            if tl.sum(final_mask.to(tl.int32)) > 0:
                bb_prev_ptr = bb + j * N + (i_idx - 1)
                bb_curr_ptr = bb + j * N + i_idx
                cc_ptr = cc + j * N + i_idx
                
                bb_prev = tl.load(bb_prev_ptr)
                cc_val_bb = tl.load(cc_ptr)
                
                result_bb = bb_prev + cc_val_bb
                tl.store(bb_curr_ptr, result_bb)

def s233_triton(aa, bb, cc):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s233_kernel[grid](
        aa, bb, cc, N, BLOCK_SIZE
    )