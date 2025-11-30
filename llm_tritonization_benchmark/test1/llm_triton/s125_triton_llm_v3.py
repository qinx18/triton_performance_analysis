import torch
import triton
import triton.language as tl

@triton.jit
def s125_kernel(flat_2d_array_ptr, aa_ptr, bb_ptr, cc_ptr, 
                LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate starting offset for this block
    offset = pid * BLOCK_SIZE
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid indices
    mask = offsets < LEN_2D * LEN_2D
    
    # Load data from 2D arrays using linear indexing
    aa_vals = tl.load(aa_ptr + offsets, mask=mask, other=0.0)
    bb_vals = tl.load(bb_ptr + offsets, mask=mask, other=0.0)
    cc_vals = tl.load(cc_ptr + offsets, mask=mask, other=0.0)
    
    # Compute: aa[i][j] + bb[i][j] * cc[i][j]
    result = aa_vals + bb_vals * cc_vals
    
    # Store result to flat array
    tl.store(flat_2d_array_ptr + offsets, result, mask=mask)

def s125_triton(flat_2d_array, aa, bb, cc):
    LEN_2D = aa.shape[0]
    total_elements = LEN_2D * LEN_2D
    
    # Choose block size
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    # Flatten 2D arrays for linear access
    aa_flat = aa.contiguous().view(-1)
    bb_flat = bb.contiguous().view(-1)
    cc_flat = cc.contiguous().view(-1)
    
    s125_kernel[grid](
        flat_2d_array, aa_flat, bb_flat, cc_flat,
        LEN_2D, BLOCK_SIZE
    )