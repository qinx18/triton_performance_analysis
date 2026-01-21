import torch
import triton
import triton.language as tl

@triton.jit
def s275_kernel(aa_ptr, bb_ptr, cc_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < n
    
    # Load aa[0][i] to check condition
    aa_0_ptrs = aa_ptr + 0 * n + i_idx
    condition_vals = tl.load(aa_0_ptrs, mask=i_mask, other=0.0)
    process_mask = (condition_vals > 0.0) & i_mask
    
    # Sequential loop over j dimension starting from 1
    for j in range(1, n):
        # Load bb[j][i]
        bb_ptrs = bb_ptr + j * n + i_idx
        bb_vals = tl.load(bb_ptrs, mask=i_mask, other=0.0)
        
        # Load cc[j][i]
        cc_ptrs = cc_ptr + j * n + i_idx
        cc_vals = tl.load(cc_ptrs, mask=i_mask, other=0.0)
        
        # Load aa[j-1][i] (previous row)
        aa_prev_ptrs = aa_ptr + (j - 1) * n + i_idx
        aa_prev_vals = tl.load(aa_prev_ptrs, mask=i_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i]
        result = aa_prev_vals + bb_vals * cc_vals
        
        # Store back to aa[j][i] only where condition is true
        aa_curr_ptrs = aa_ptr + j * n + i_idx
        
        # Load current aa[j][i] values for conditional update
        aa_curr_vals = tl.load(aa_curr_ptrs, mask=i_mask, other=0.0)
        final_vals = tl.where(process_mask, result, aa_curr_vals)
        tl.store(aa_curr_ptrs, final_vals, mask=i_mask)

def s275_triton(aa, bb, cc):
    n = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s275_kernel[grid](
        aa, bb, cc, n,
        BLOCK_SIZE=BLOCK_SIZE
    )