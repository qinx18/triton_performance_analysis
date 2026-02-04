import triton
import triton.language as tl
import torch

@triton.jit
def s276_kernel_part1(a_ptr, b_ptr, c_ptr, n_elements, mid, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    condition_mask = (offsets + 1) < mid
    
    combined_mask = mask & condition_mask
    
    a_vals = tl.load(a_ptr + offsets, mask=combined_mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=combined_mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=combined_mask, other=0.0)
    
    result = a_vals + b_vals * c_vals
    
    tl.store(a_ptr + offsets, result, mask=combined_mask)

@triton.jit
def s276_kernel_part2(a_ptr, b_ptr, d_ptr, n_elements, mid, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    condition_mask = (offsets + 1) >= mid
    
    combined_mask = mask & condition_mask
    
    a_vals = tl.load(a_ptr + offsets, mask=combined_mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=combined_mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=combined_mask, other=0.0)
    
    result = a_vals + b_vals * d_vals
    
    tl.store(a_ptr + offsets, result, mask=combined_mask)

def s276_triton(a, b, c, d, mid):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s276_kernel_part1[grid](
        a, b, c, n_elements, mid, BLOCK_SIZE
    )
    
    s276_kernel_part2[grid](
        a, b, d, n_elements, mid, BLOCK_SIZE
    )