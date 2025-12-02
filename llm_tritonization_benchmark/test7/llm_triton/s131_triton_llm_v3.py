import torch
import triton
import triton.language as tl

@triton.jit
def s131_kernel(
    a_ptr,
    a_copy_ptr,
    b_ptr,
    m,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    read_offsets = i_offsets + m
    
    mask = i_offsets < n_elements
    read_mask = read_offsets < (n_elements + m)
    
    a_vals = tl.load(a_copy_ptr + read_offsets, mask=read_mask)
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    
    result = a_vals + b_vals
    
    tl.store(a_ptr + i_offsets, result, mask=mask)

def s131_triton(a, b, m):
    n_elements = a.shape[0] - 1
    
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s131_kernel[grid](
        a,
        a_copy,
        b,
        m,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )