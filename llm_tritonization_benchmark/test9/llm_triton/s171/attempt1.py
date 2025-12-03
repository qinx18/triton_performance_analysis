import triton
import triton.language as tl
import torch

@triton.jit
def s171_kernel(a_ptr, b_ptr, inc, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    mask = i_offsets < n_elements
    
    # Load b[i]
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    
    # Calculate a indices: i * inc
    a_indices = i_offsets * inc
    
    # Load a[i * inc]
    a_vals = tl.load(a_ptr + a_indices, mask=mask)
    
    # Compute a[i * inc] += b[i]
    result = a_vals + b_vals
    
    # Store back to a[i * inc]
    tl.store(a_ptr + a_indices, result, mask=mask)

def s171_triton(a, b, inc):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s171_kernel[grid](
        a, b, inc, n_elements, BLOCK_SIZE
    )