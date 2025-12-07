import triton
import triton.language as tl
import torch

@triton.jit
def s113_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Adjust offsets to start from index 1
    offsets = offsets + 1
    mask = offsets < n_elements
    
    # Load a[0] once (scalar broadcast)
    a0 = tl.load(a_ptr)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] = a[0] + b[i]
    result = a0 + b_vals
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s113_triton(a, b):
    n_elements = a.shape[0]
    # Process elements from index 1 to n_elements-1
    elements_to_process = n_elements - 1
    
    if elements_to_process <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(elements_to_process, BLOCK_SIZE),)
    
    s113_kernel[grid](a, b, n_elements, BLOCK_SIZE)