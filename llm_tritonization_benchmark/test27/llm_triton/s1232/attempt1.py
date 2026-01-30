import triton
import triton.language as tl
import torch

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    
    for j in range(len_2d):
        # Only process elements where i >= j (triangular loop condition)
        mask = (i_idx < len_2d) & (i_idx >= j)
        
        # Calculate memory offsets
        bb_offsets = i_idx * len_2d + j
        cc_offsets = i_idx * len_2d + j
        aa_offsets = i_idx * len_2d + j
        
        # Load data
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask)
        cc_vals = tl.load(cc_ptr + cc_offsets, mask=mask)
        
        # Compute
        aa_vals = bb_vals + cc_vals
        
        # Store result
        tl.store(aa_ptr + aa_offsets, aa_vals, mask=mask)

def s1232_triton(aa, bb, cc, len_2d):
    BLOCK_SIZE = 128
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s1232_kernel[grid](
        aa, bb, cc,
        len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )