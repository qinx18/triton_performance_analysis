import triton
import triton.language as tl
import torch

@triton.jit
def s175_kernel(a_ptr, b_ptr, inc, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Convert linear indices to strided indices
    strided_indices = offsets * inc
    mask = strided_indices < (n_elements - 1)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + strided_indices, mask=mask)
    
    # Load a[i + inc] values
    read_indices = strided_indices + inc
    read_mask = mask & (read_indices < n_elements)
    a_read_vals = tl.load(a_ptr + read_indices, mask=read_mask)
    
    # Compute result
    result = a_read_vals + b_vals
    
    # Store to a[i]
    tl.store(a_ptr + strided_indices, result, mask=mask)

def s175_triton(a, b, inc):
    n_elements = a.shape[0]
    
    # Calculate number of elements that will be processed
    num_iterations = (n_elements - 1 + inc - 1) // inc
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_iterations, BLOCK_SIZE),)
    
    s175_kernel[grid](
        a, b, inc, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )