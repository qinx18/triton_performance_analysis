import torch
import triton
import triton.language as tl

@triton.jit
def s151s_kernel(
    a_ptr,
    b_ptr, 
    n_elements,
    k_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load from a and b
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Simple computation with k_val
    result = a_vals + b_vals * k_val
    
    # Store back to a
    tl.store(a_ptr + indices, result, mask=mask)

@triton.jit
def s151_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Call s151s with k=1
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # s151s operation with k=1
    result = a_vals + b_vals * 1.0
    
    # Store result back to a
    tl.store(a_ptr + indices, result, mask=mask)

def s151_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s151_kernel[grid](
        a,
        b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a