import triton
import triton.language as tl
import torch

@triton.jit
def s162_kernel(a_ptr, b_ptr, c_ptr, k, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load b[i] and c[i]
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    
    # Load a[i + k] with bounds checking
    read_indices = indices + k
    read_mask = mask & (read_indices < (n_elements + 1))  # Original array size
    a_vals = tl.load(a_ptr + read_indices, mask=read_mask)
    
    # Compute a[i] = a[i + k] + b[i] * c[i]
    result = a_vals + b_vals * c_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s162_triton(a, b, c, k):
    if k <= 0:
        return
    
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s162_kernel[grid](
        a, b, c, k, n_elements, BLOCK_SIZE
    )