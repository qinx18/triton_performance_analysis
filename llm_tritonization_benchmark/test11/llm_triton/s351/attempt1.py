import torch
import triton
import triton.language as tl

@triton.jit
def s351_kernel(a_ptr, b_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for group_start in range(block_start, n_elements, BLOCK_SIZE * tl.num_programs(0)):
        for offset in range(0, 5):
            indices = group_start + offsets * 5 + offset
            mask = indices < n_elements
            
            a_vals = tl.load(a_ptr + indices, mask=mask)
            b_vals = tl.load(b_ptr + indices, mask=mask)
            
            result = a_vals + alpha * b_vals
            tl.store(a_ptr + indices, result, mask=mask)

def s351_triton(a, b, c):
    n_elements = a.shape[0]
    alpha = c[0].item()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE * 5),)
    
    s351_kernel[grid](
        a, b, alpha, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )