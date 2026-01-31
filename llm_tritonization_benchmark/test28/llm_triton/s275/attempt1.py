import triton
import triton.language as tl
import torch

@triton.jit
def s275_kernel(aa_ptr, bb_ptr, cc_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < len_2d
    
    # Load aa[0][i] for condition check
    aa_0_ptr = aa_ptr + i_idx
    aa_0_vals = tl.load(aa_0_ptr, mask=i_mask, other=0.0)
    condition_mask = aa_0_vals > 0.0
    
    # Combined mask for valid indices and condition
    valid_mask = i_mask & condition_mask
    
    # Sequential loop over j dimension
    for j in range(1, len_2d):
        # Load aa[j-1][i]
        aa_prev_ptr = aa_ptr + (j - 1) * len_2d + i_idx
        aa_prev_vals = tl.load(aa_prev_ptr, mask=valid_mask, other=0.0)
        
        # Load bb[j][i]
        bb_ptr_j = bb_ptr + j * len_2d + i_idx
        bb_vals = tl.load(bb_ptr_j, mask=valid_mask, other=0.0)
        
        # Load cc[j][i]
        cc_ptr_j = cc_ptr + j * len_2d + i_idx
        cc_vals = tl.load(cc_ptr_j, mask=valid_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i]
        result = aa_prev_vals + bb_vals * cc_vals
        
        # Store aa[j][i]
        aa_curr_ptr = aa_ptr + j * len_2d + i_idx
        tl.store(aa_curr_ptr, result, mask=valid_mask)

def s275_triton(aa, bb, cc, len_2d):
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s275_kernel[grid](
        aa, bb, cc, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )