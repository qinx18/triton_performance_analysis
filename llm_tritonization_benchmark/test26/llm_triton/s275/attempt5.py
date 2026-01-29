import torch
import triton
import triton.language as tl

@triton.jit
def s275_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < LEN_2D
    
    # Check condition for each i: aa[0][i] > 0
    aa_0_ptrs = aa_ptr + 0 * LEN_2D + i_idx
    aa_0_vals = tl.load(aa_0_ptrs, mask=i_mask, other=0.0)
    condition_mask = (aa_0_vals > 0.0) & i_mask
    
    # Sequential loop over j dimension
    for j in range(1, LEN_2D):
        # Load aa[j-1][i] (previous row)
        aa_prev_ptrs = aa_ptr + (j - 1) * LEN_2D + i_idx
        aa_prev_vals = tl.load(aa_prev_ptrs, mask=i_mask, other=0.0)
        
        # Load bb[j][i]
        bb_ptrs = bb_ptr + j * LEN_2D + i_idx
        bb_vals = tl.load(bb_ptrs, mask=i_mask, other=0.0)
        
        # Load cc[j][i]
        cc_ptrs = cc_ptr + j * LEN_2D + i_idx
        cc_vals = tl.load(cc_ptrs, mask=i_mask, other=0.0)
        
        # Load current aa[j][i]
        aa_curr_ptrs = aa_ptr + j * LEN_2D + i_idx
        aa_curr_vals = tl.load(aa_curr_ptrs, mask=i_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i]
        # Only update where condition is true, otherwise keep original value
        result = tl.where(condition_mask, aa_prev_vals + bb_vals * cc_vals, aa_curr_vals)
        
        # Store back to aa[j][i]
        tl.store(aa_curr_ptrs, result, mask=i_mask)

def s275_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s275_kernel[grid](
        aa, bb, cc,
        LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )