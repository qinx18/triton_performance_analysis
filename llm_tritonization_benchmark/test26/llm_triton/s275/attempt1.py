import triton
import triton.language as tl
import torch

@triton.jit
def s275_kernel(aa_ptr, bb_ptr, cc_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < len_2d
    
    # Load condition values (aa[0][i])
    cond_ptrs = aa_ptr + i_idx
    cond_vals = tl.load(cond_ptrs, mask=i_mask, other=0.0)
    cond_mask = cond_vals > 0.0
    
    # Combined mask for both bounds and condition
    active_mask = i_mask & cond_mask
    
    # Sequential loop over j dimension (dependencies require ordering)
    for j in range(1, len_2d):
        # Load aa[j-1][i] (previous row)
        prev_ptrs = aa_ptr + (j - 1) * len_2d + i_idx
        prev_vals = tl.load(prev_ptrs, mask=active_mask, other=0.0)
        
        # Load bb[j][i]
        bb_ptrs = bb_ptr + j * len_2d + i_idx
        bb_vals = tl.load(bb_ptrs, mask=active_mask, other=0.0)
        
        # Load cc[j][i]
        cc_ptrs = cc_ptr + j * len_2d + i_idx
        cc_vals = tl.load(cc_ptrs, mask=active_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i]
        result = prev_vals + bb_vals * cc_vals
        
        # Store aa[j][i]
        curr_ptrs = aa_ptr + j * len_2d + i_idx
        tl.store(curr_ptrs, result, mask=active_mask)

def s275_triton(aa, bb, cc, len_2d):
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s275_kernel[grid](
        aa, bb, cc, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )