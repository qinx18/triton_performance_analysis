import triton
import triton.language as tl
import torch

@triton.jit
def s275_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = pid * BLOCK_SIZE + i_offsets
    i_mask = i_indices < LEN_2D
    
    # Check condition aa[0][i] > 0
    aa_0_ptrs = aa_ptr + i_indices
    aa_0_vals = tl.load(aa_0_ptrs, mask=i_mask, other=0.0)
    condition_mask = (aa_0_vals > 0.0) & i_mask
    
    # Sequential loop over j
    for j in range(1, LEN_2D):
        # Load aa[j-1][i]
        aa_prev_ptrs = aa_ptr + (j - 1) * LEN_2D + i_indices
        aa_prev_vals = tl.load(aa_prev_ptrs, mask=condition_mask, other=0.0)
        
        # Load bb[j][i] and cc[j][i]
        bb_ptrs = bb_ptr + j * LEN_2D + i_indices
        cc_ptrs = cc_ptr + j * LEN_2D + i_indices
        bb_vals = tl.load(bb_ptrs, mask=condition_mask, other=0.0)
        cc_vals = tl.load(cc_ptrs, mask=condition_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i]
        new_vals = aa_prev_vals + bb_vals * cc_vals
        
        # Store aa[j][i]
        aa_curr_ptrs = aa_ptr + j * LEN_2D + i_indices
        tl.store(aa_curr_ptrs, new_vals, mask=condition_mask)

def s275_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = min(256, LEN_2D)
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s275_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )