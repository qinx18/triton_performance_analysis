import torch
import triton
import triton.language as tl

@triton.jit
def s151s_kernel(a_ptr, b_ptr, s, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load values from arrays
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    
    # Perform computation: a[i] = a[i] + b[i] * s
    result = a_vals + b_vals * s
    
    # Store result back to a
    tl.store(a_ptr + idx, result, mask=mask)

@triton.jit
def s151_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Call s151s with parameters a, b, 1
    # Load values from arrays
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    
    # s151s computation: a[i] = a[i] + b[i] * 1
    result = a_vals + b_vals * 1.0
    
    # Store result back to a
    tl.store(a_ptr + idx, result, mask=mask)

def s151_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s151_kernel[grid](a, b, n_elements, BLOCK_SIZE)