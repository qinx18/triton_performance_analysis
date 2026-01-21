import triton
import triton.language as tl
import torch

@triton.jit
def s1111_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    result = c_vals * b_vals + d_vals * b_vals + c_vals * c_vals + d_vals * b_vals + d_vals * c_vals
    
    output_offsets = 2 * offsets
    output_mask = offsets < n
    
    tl.store(a_ptr + output_offsets, result, mask=output_mask)

def s1111_triton(a, b, c, d):
    n = a.shape[0] // 2
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s1111_kernel[grid](a, b, c, d, n, BLOCK_SIZE=BLOCK_SIZE)