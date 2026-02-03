import triton
import triton.language as tl
import torch

@triton.jit
def s4121_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask)
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask)
    
    # Compute f(b[i], c[i]) = b[i] * c[i]
    f_result = b_vals * c_vals
    
    # Update a[i] += f(b[i], c[i])
    result = a_vals + f_result
    
    # Store result
    tl.store(a_ptr + block_start + offsets, result, mask=mask)

def s4121_triton(a, b, c):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s4121_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)