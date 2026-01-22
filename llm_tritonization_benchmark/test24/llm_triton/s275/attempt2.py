import torch
import triton
import triton.language as tl

@triton.jit
def s275_kernel(aa_ptr, bb_ptr, cc_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    
    # Check condition aa[0][i] > 0
    mask_i = i_idx < len_2d
    aa_0_ptr = aa_ptr + i_idx
    condition_vals = tl.load(aa_0_ptr, mask=mask_i, other=0.0)
    condition_mask = condition_vals > 0.0
    
    # Sequential loop over j dimension
    for j in range(1, len_2d):
        # Load aa[j-1][i]
        aa_prev_ptr = aa_ptr + (j - 1) * len_2d + i_idx
        aa_prev = tl.load(aa_prev_ptr, mask=mask_i & condition_mask, other=0.0)
        
        # Load bb[j][i]
        bb_ptr_offset = bb_ptr + j * len_2d + i_idx
        bb_val = tl.load(bb_ptr_offset, mask=mask_i & condition_mask, other=0.0)
        
        # Load cc[j][i]
        cc_ptr_offset = cc_ptr + j * len_2d + i_idx
        cc_val = tl.load(cc_ptr_offset, mask=mask_i & condition_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i]
        result = aa_prev + bb_val * cc_val
        
        # Store result
        aa_curr_ptr = aa_ptr + j * len_2d + i_idx
        tl.store(aa_curr_ptr, result, mask=mask_i & condition_mask)

def s275_triton(aa, bb, cc):
    len_2d = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s275_kernel[grid](
        aa, bb, cc,
        len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )