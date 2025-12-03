import torch
import triton
import triton.language as tl

@triton.jit
def s162_kernel(a_ptr, b_ptr, c_ptr, n_elements, k, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load b[i] and c[i]
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    
    # Load a[i + k]
    shifted_indices = indices + k
    shifted_mask = mask & (shifted_indices < (n_elements + k))
    a_shifted_vals = tl.load(a_ptr + shifted_indices, mask=shifted_mask)
    
    # Compute a[i] = a[i + k] + b[i] * c[i]
    result = a_shifted_vals + b_vals * c_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s162_triton(a, b, c, k):
    if k <= 0:
        return
        
    n_elements = a.shape[0] - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s162_kernel[grid](a, b, c, n_elements, k, BLOCK_SIZE=BLOCK_SIZE)