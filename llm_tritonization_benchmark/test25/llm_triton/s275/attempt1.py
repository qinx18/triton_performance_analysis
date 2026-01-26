import triton
import triton.language as tl
import torch

@triton.jit
def s275_kernel(aa_ptr, bb_ptr, cc_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Define offsets once at the start
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < len_2d
    
    # Check condition aa[0][i] > 0
    aa_0_ptr = aa_ptr + i_idx
    aa_0_vals = tl.load(aa_0_ptr, mask=i_mask, other=0.0)
    condition_mask = (aa_0_vals > 0.0) & i_mask
    
    # Sequential loop over j dimension inside kernel
    for j in range(1, len_2d):
        if tl.any(condition_mask):
            # Calculate pointers for current and previous j
            aa_j_ptr = aa_ptr + j * len_2d + i_idx
            aa_j_prev_ptr = aa_ptr + (j - 1) * len_2d + i_idx
            bb_j_ptr = bb_ptr + j * len_2d + i_idx
            cc_j_ptr = cc_ptr + j * len_2d + i_idx
            
            # Load values
            aa_j_prev = tl.load(aa_j_prev_ptr, mask=condition_mask, other=0.0)
            bb_j = tl.load(bb_j_ptr, mask=condition_mask, other=0.0)
            cc_j = tl.load(cc_j_ptr, mask=condition_mask, other=0.0)
            
            # Compute: aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i]
            result = aa_j_prev + bb_j * cc_j
            
            # Store result
            tl.store(aa_j_ptr, result, mask=condition_mask)

def s275_triton(aa, bb, cc, len_2d):
    BLOCK_SIZE = 256
    
    # Single kernel launch for i-parallel dimension
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s275_kernel[grid](
        aa, bb, cc,
        len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )