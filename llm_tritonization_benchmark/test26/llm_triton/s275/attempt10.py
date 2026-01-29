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
    
    # Initialize current values with aa[0][i]
    current_vals = cond_vals
    
    # For each j from 1 to N-1, update aa[j][i] sequentially
    for j in range(1, N):
        # Current row pointers
        curr_aa_ptr = aa_ptr + j * N + i_idx
        bb_ptr_j = bb_ptr + j * N + i_idx
        cc_ptr_j = cc_ptr + j * N + i_idx
        
        # Load values
        bb_vals = tl.load(bb_ptr_j, mask=i_mask, other=0.0)
        cc_vals = tl.load(cc_ptr_j, mask=i_mask, other=0.0)
        
        # Compute: aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i]
        new_vals = current_vals + bb_vals * cc_vals
        
        # Only store where condition is true AND index is valid
        tl.store(curr_aa_ptr, new_vals, mask=cond_mask)
        
        # Update current values for next iteration
        current_vals = tl.where(cond_mask, new_vals, current_vals)

def s275_triton(aa, bb, cc):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s275_kernel[grid](
        aa, bb, cc, N, BLOCK_SIZE
    )