import torch
import triton
import triton.language as tl

@triton.jit
def s2101_kernel(
    aa_ptr, bb_ptr, cc_ptr,
    len_2d,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, len_2d, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < len_2d
        
        # Calculate diagonal indices: i * len_2d + i for aa[i][i]
        diag_indices = current_offsets * len_2d + current_offsets
        
        # Load diagonal elements
        aa_vals = tl.load(aa_ptr + diag_indices, mask=mask)
        bb_vals = tl.load(bb_ptr + diag_indices, mask=mask)
        cc_vals = tl.load(cc_ptr + diag_indices, mask=mask)
        
        # Compute: aa[i][i] += bb[i][i] * cc[i][i]
        result = aa_vals + bb_vals * cc_vals
        
        # Store result
        tl.store(aa_ptr + diag_indices, result, mask=mask)

def s2101_triton(aa, bb, cc):
    len_2d = aa.shape[0]
    
    BLOCK_SIZE = 64
    grid = (1,)
    
    s2101_kernel[grid](
        aa, bb, cc,
        len_2d,
        BLOCK_SIZE=BLOCK_SIZE,
    )