import torch
import triton
import triton.language as tl

@triton.jit
def s275_kernel(aa_ptr, bb_ptr, cc_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < len_2d
    
    # Check condition aa[0][i] > 0
    aa_0_ptr = aa_ptr + i_offsets
    aa_0_vals = tl.load(aa_0_ptr, mask=i_mask, other=0.0)
    condition = aa_0_vals > 0.0
    
    # Sequential loop over j dimension
    for j in range(1, len_2d):
        # Load aa[j-1][i]
        aa_prev_ptr = aa_ptr + (j - 1) * len_2d + i_offsets
        aa_prev_vals = tl.load(aa_prev_ptr, mask=i_mask & condition, other=0.0)
        
        # Load bb[j][i]
        bb_ptr_offset = bb_ptr + j * len_2d + i_offsets
        bb_vals = tl.load(bb_ptr_offset, mask=i_mask & condition, other=0.0)
        
        # Load cc[j][i]
        cc_ptr_offset = cc_ptr + j * len_2d + i_offsets
        cc_vals = tl.load(cc_ptr_offset, mask=i_mask & condition, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i]
        result = aa_prev_vals + bb_vals * cc_vals
        
        # Store result
        aa_curr_ptr = aa_ptr + j * len_2d + i_offsets
        tl.store(aa_curr_ptr, result, mask=i_mask & condition)

def s275_triton(aa, bb, cc, len_2d):
    BLOCK_SIZE = 256
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s275_kernel[grid](
        aa, bb, cc, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )