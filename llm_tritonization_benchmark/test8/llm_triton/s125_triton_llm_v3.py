import torch
import triton
import triton.language as tl

@triton.jit
def s125_kernel(aa_ptr, bb_ptr, cc_ptr, flat_2d_array_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    indices = block_start + offsets
    mask = indices < n_elements
    
    aa_vals = tl.load(aa_ptr + indices, mask=mask)
    bb_vals = tl.load(bb_ptr + indices, mask=mask)
    cc_vals = tl.load(cc_ptr + indices, mask=mask)
    
    result = aa_vals + bb_vals * cc_vals
    
    tl.store(flat_2d_array_ptr + indices, result, mask=mask)

def s125_triton(aa, bb, cc, flat_2d_array):
    n_elements = aa.numel()
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s125_kernel[grid](
        aa, bb, cc, flat_2d_array,
        n_elements,
        BLOCK_SIZE
    )