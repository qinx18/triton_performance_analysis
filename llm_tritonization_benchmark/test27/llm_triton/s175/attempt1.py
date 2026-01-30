import triton
import triton.language as tl
import torch

@triton.jit
def s175_kernel(a_ptr, a_copy_ptr, b_ptr, inc, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets * inc
    
    mask = i_offsets < (n - 1)
    
    # Load from read-only copy for a[i + inc] and original b for b[i]
    a_read_offsets = i_offsets + inc
    a_read_mask = mask & (a_read_offsets < n)
    
    a_vals = tl.load(a_copy_ptr + a_read_offsets, mask=a_read_mask, other=0.0)
    b_vals = tl.load(b_ptr + i_offsets, mask=mask, other=0.0)
    
    result = a_vals + b_vals
    
    # Store to original array
    tl.store(a_ptr + i_offsets, result, mask=mask)

def s175_triton(a, b, inc):
    n = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Calculate number of iterations
    num_elements = (n - 1 + inc - 1) // inc
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    s175_kernel[grid](
        a, a_copy, b, inc, n, BLOCK_SIZE
    )