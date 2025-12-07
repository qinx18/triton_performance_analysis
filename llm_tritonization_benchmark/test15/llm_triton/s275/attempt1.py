import torch
import triton
import triton.language as tl

@triton.jit
def s275_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    mask_i = i_idx < LEN_2D
    
    # Load condition values aa[0][i]
    aa_0_ptrs = aa_ptr + 0 * LEN_2D + i_idx
    condition_vals = tl.load(aa_0_ptrs, mask=mask_i, other=0.0)
    condition_mask = condition_vals > 0.0
    
    # Sequential loop over j (dependency requires ordering)
    for j in range(1, LEN_2D):
        # Only process if condition is true
        combined_mask = mask_i & condition_mask
        
        if tl.any(combined_mask):
            # Load aa[j-1][i]
            aa_prev_ptrs = aa_ptr + (j - 1) * LEN_2D + i_idx
            aa_prev = tl.load(aa_prev_ptrs, mask=combined_mask, other=0.0)
            
            # Load bb[j][i]
            bb_ptrs = bb_ptr + j * LEN_2D + i_idx
            bb_vals = tl.load(bb_ptrs, mask=combined_mask, other=0.0)
            
            # Load cc[j][i]
            cc_ptrs = cc_ptr + j * LEN_2D + i_idx
            cc_vals = tl.load(cc_ptrs, mask=combined_mask, other=0.0)
            
            # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i]
            result = aa_prev + bb_vals * cc_vals
            
            # Store aa[j][i]
            aa_curr_ptrs = aa_ptr + j * LEN_2D + i_idx
            tl.store(aa_curr_ptrs, result, mask=combined_mask)

def s275_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s275_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )