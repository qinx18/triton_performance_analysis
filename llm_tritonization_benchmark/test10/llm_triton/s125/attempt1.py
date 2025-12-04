import torch
import triton
import triton.language as tl

@triton.jit
def s125_kernel(aa_ptr, bb_ptr, cc_ptr, flat_2d_array_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    indices = block_start + offsets
    
    mask = indices < LEN_2D * LEN_2D
    
    aa_vals = tl.load(aa_ptr + indices, mask=mask)
    bb_vals = tl.load(bb_ptr + indices, mask=mask)
    cc_vals = tl.load(cc_ptr + indices, mask=mask)
    
    result = aa_vals + bb_vals * cc_vals
    
    tl.store(flat_2d_array_ptr + indices, result, mask=mask)

def s125_triton(aa, bb, cc, flat_2d_array):
    LEN_2D = aa.shape[0]
    total_elements = LEN_2D * LEN_2D
    
    aa_flat = aa.flatten()
    bb_flat = bb.flatten()
    cc_flat = cc.flatten()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    s125_kernel[grid](
        aa_flat, bb_flat, cc_flat, flat_2d_array,
        LEN_2D, BLOCK_SIZE
    )