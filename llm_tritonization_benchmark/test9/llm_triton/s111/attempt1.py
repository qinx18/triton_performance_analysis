import torch
import triton
import triton.language as tl

@triton.jit
def s111_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each block processes consecutive odd indices
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    # Map to odd indices: 1, 3, 5, 7, ...
    odd_indices = 1 + 2 * (block_start + offsets)
    prev_indices = odd_indices - 1
    
    mask = odd_indices < n_elements
    
    # Load a[i-1] and b[i]
    a_prev = tl.load(a_ptr + prev_indices, mask=mask)
    b_vals = tl.load(b_ptr + odd_indices, mask=mask)
    
    # Compute a[i] = a[i-1] + b[i]
    result = a_prev + b_vals
    
    # Store back to a[i]
    tl.store(a_ptr + odd_indices, result, mask=mask)

def s111_triton(a, b):
    n_elements = a.shape[0]
    # Number of odd indices to process: (1, 3, 5, ..., n-1 or n-2)
    num_odd_indices = (n_elements - 1 + 1) // 2  # ceiling division for odd indices starting at 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_odd_indices, BLOCK_SIZE),)
    
    s111_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )