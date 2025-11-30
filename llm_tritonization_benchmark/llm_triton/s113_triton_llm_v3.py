import triton
import triton.language as tl
import torch

@triton.jit
def s113_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load a[0] (scalar broadcast)
    a0 = tl.load(a_ptr)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] = a[0] + b[i]
    result = a0 + b_vals
    
    # Store result to a[i]
    tl.store(a_ptr + offsets, result, mask=mask)

def s113_triton(a, b):
    n_elements = a.shape[0] - 1  # Process elements 1 to LEN_1D-1
    
    if n_elements <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel starting from index 1
    s113_kernel[grid](
        a[1:],  # Start from a[1]
        b[1:],  # Start from b[1]
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )