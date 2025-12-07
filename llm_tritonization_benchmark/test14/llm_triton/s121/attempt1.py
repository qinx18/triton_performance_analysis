import triton
import triton.language as tl
import torch

@triton.jit
def s121_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load a[j] where j = i + 1
    a_j = tl.load(a_ptr + idx + 1, mask=mask)
    
    # Load b[i]
    b_i = tl.load(b_ptr + idx, mask=mask)
    
    # Compute a[i] = a[j] + b[i]
    result = a_j + b_i
    
    # Store result to a[i]
    tl.store(a_ptr + idx, result, mask=mask)

def s121_triton(a, b):
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s121_kernel[grid](a, b, n_elements, BLOCK_SIZE)
    
    return a