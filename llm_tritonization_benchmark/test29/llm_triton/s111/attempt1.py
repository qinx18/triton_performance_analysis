import triton
import triton.language as tl
import torch

@triton.jit
def s111_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    offsets = tl.arange(0, BLOCK_SIZE)
    # Map to original loop indices: i = 1, 3, 5, 7, ...
    indices = 1 + 2 * (pid * BLOCK_SIZE + offsets)
    
    mask = indices < n
    
    # Load b[i]
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Load a[i-1]
    prev_indices = indices - 1
    a_prev_vals = tl.load(a_ptr + prev_indices, mask=mask)
    
    # Compute a[i] = a[i-1] + b[i]
    result = a_prev_vals + b_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s111_triton(a, b):
    n = a.shape[0]
    
    # Calculate number of elements to process (odd indices from 1 to n-1)
    num_elements = (n - 1 + 1) // 2  # Number of odd indices from 1 to n-1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    s111_kernel[grid](a, b, n, BLOCK_SIZE)