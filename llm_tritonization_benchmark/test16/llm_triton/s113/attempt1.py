import triton
import triton.language as tl
import torch

@triton.jit
def s113_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets + 1  # Start from index 1
    
    mask = indices < n_elements
    
    # Load a[0] (scalar broadcast)
    a_0 = tl.load(a_ptr)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute a[i] = a[0] + b[i]
    result = a_0 + b_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s113_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Number of elements to process (excluding a[0])
    n_to_process = n_elements - 1
    
    if n_to_process > 0:
        grid = (triton.cdiv(n_to_process, BLOCK_SIZE),)
        s113_kernel[grid](a, b, n_elements, BLOCK_SIZE)
    
    return a