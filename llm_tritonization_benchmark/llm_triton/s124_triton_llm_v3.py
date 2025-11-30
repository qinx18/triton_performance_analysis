import triton
import triton.language as tl
import torch

@triton.jit
def s124_kernel(b_ptr, c_ptr, d_ptr, e_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    e = tl.load(e_ptr + offsets, mask=mask)
    
    # Compute d[i] * e[i] for both branches
    de_product = d * e
    
    # Apply conditional logic: select b or c based on condition
    condition = b > 0.0
    selected = tl.where(condition, b, c)
    
    # Compute final result
    result = selected + de_product
    
    # Store result (output indices are same as input indices since j increments for both branches)
    tl.store(a_ptr + offsets, result, mask=mask)

def s124_triton(a, b, c, d, e):
    n_elements = b.numel()
    
    # Choose block size
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s124_kernel[grid](
        b, c, d, e, a,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )