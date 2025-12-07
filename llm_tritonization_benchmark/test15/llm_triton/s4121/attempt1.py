import torch
import triton
import triton.language as tl

@triton.jit
def s4121_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    
    # Compute f(b[i], c[i]) = b[i] * c[i]
    f_result = b_vals * c_vals
    
    # Update a[i] += f(b[i], c[i])
    result = a_vals + f_result
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s4121_triton(a, b, c):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s4121_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)