import triton
import triton.language as tl
import torch

@triton.jit
def s275_kernel(aa_ptr, bb_ptr, cc_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < N
    
    # Load the condition values aa[0][i]
    cond_ptr = aa_ptr + i_idx
    cond_vals = tl.load(cond_ptr, mask=i_mask, other=0.0)
    cond_mask = (cond_vals > 0.0) & i_mask
    
    # For each j from 1 to N-1, update aa[j][i] sequentially
    for j in range(1, N):
        # Current and previous row pointers
        curr_aa_ptr = aa_ptr + j * N + i_idx
        prev_aa_ptr = aa_ptr + (j - 1) * N + i_idx
        bb_ptr_j = bb_ptr + j * N + i_idx
        cc_ptr_j = cc_ptr + j * N + i_idx
        
        # Load values
        prev_aa_vals = tl.load(prev_aa_ptr, mask=i_mask, other=0.0)
        bb_vals = tl.load(bb_ptr_j, mask=i_mask, other=0.0)
        cc_vals = tl.load(cc_ptr_j, mask=i_mask, other=0.0)
        
        # Compute: aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i]
        new_vals = prev_aa_vals + bb_vals * cc_vals
        
        # Only store where condition is true AND index is valid
        store_mask = cond_mask
        tl.store(curr_aa_ptr, new_vals, mask=store_mask)

def s275_triton(aa, bb, cc):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s275_kernel[grid](
        aa, bb, cc, N, BLOCK_SIZE
    )