import torch
import triton
import triton.language as tl

@triton.jit
def s1111_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    mask = (block_start + offsets) < n_elements
    input_offsets = block_start + offsets
    output_offsets = 2 * (block_start + offsets)
    
    b_vals = tl.load(b_ptr + input_offsets, mask=mask)
    c_vals = tl.load(c_ptr + input_offsets, mask=mask)
    d_vals = tl.load(d_ptr + input_offsets, mask=mask)
    
    result = c_vals * b_vals + d_vals * b_vals + c_vals * c_vals + d_vals * b_vals + d_vals * c_vals
    
    tl.store(a_ptr + output_offsets, result, mask=mask)

def s1111_triton(a, b, c, d):
    n_elements = a.shape[0] // 2
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1111_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE,
    )