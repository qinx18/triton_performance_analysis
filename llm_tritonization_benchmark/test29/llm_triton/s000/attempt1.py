import triton
import triton.language as tl
import torch

@triton.jit
def s000_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    b_vals = tl.load(b_ptr + idx, mask=mask)
    result = b_vals + 1
    tl.store(a_ptr + idx, result, mask=mask)

def s000_triton(a, b):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s000_kernel[grid](
        a,
        b, 
        n_elements,
        BLOCK_SIZE,
    )