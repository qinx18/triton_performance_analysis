import triton
import triton.language as tl
import torch

@triton.jit
def s1112_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Reverse the offsets to implement the reverse loop
    reversed_offsets = n_elements - 1 - offsets
    reversed_mask = reversed_offsets >= 0
    final_mask = mask & reversed_mask
    
    b_vals = tl.load(b_ptr + reversed_offsets, mask=final_mask)
    result = b_vals + 1.0
    tl.store(a_ptr + reversed_offsets, result, mask=final_mask)

def s1112_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1112_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )