import triton
import triton.language as tl
import torch

@triton.jit
def s2275_kernel(
    aa_ptr, bb_ptr, cc_ptr, a_ptr, b_ptr, c_ptr, d_ptr,
    len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < len_2d
    
    for i in range(len_2d):
        # 2D array indexing: aa[j][i] means j*len_2d + i
        aa_idx = j_idx * len_2d + i
        bb_idx = j_idx * len_2d + i
        cc_idx = j_idx * len_2d + i
        
        aa_vals = tl.load(aa_ptr + aa_idx, mask=j_mask, other=0.0)
        bb_vals = tl.load(bb_ptr + bb_idx, mask=j_mask, other=0.0)
        cc_vals = tl.load(cc_ptr + cc_idx, mask=j_mask, other=0.0)
        
        result = aa_vals + bb_vals * cc_vals
        tl.store(aa_ptr + aa_idx, result, mask=j_mask)

@triton.jit 
def s2275_1d_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < len_2d
    
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    
    result = b_vals + c_vals * d_vals
    tl.store(a_ptr + offsets, result, mask=mask)

def s2275_triton(a, aa, b, bb, c, cc, d, len_2d):
    BLOCK_SIZE = 256
    
    # First kernel for 2D arrays
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    s2275_kernel[grid](
        aa, bb, cc, a, b, c, d,
        len_2d, BLOCK_SIZE
    )
    
    # Second kernel for 1D arrays
    s2275_1d_kernel[grid](
        a, b, c, d, len_2d, BLOCK_SIZE
    )