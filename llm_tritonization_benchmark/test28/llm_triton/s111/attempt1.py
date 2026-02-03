import triton
import triton.language as tl
import torch

@triton.jit
def s111_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Map to original loop indices: i = 1, 3, 5, ...
    # Thread 0 -> i=1, Thread 1 -> i=3, etc.
    original_indices = 1 + 2 * (pid * BLOCK_SIZE + offsets)
    
    # Mask for valid indices
    mask = original_indices < n_elements
    
    # Load a[i-1] and b[i]
    a_prev_indices = original_indices - 1
    a_prev = tl.load(a_ptr + a_prev_indices, mask=mask)
    b_vals = tl.load(b_ptr + original_indices, mask=mask)
    
    # Compute a[i] = a[i-1] + b[i]
    result = a_prev + b_vals
    
    # Store back to a[i]
    tl.store(a_ptr + original_indices, result, mask=mask)

def s111_triton(a, b):
    n_elements = a.shape[0]
    
    # Calculate number of iterations in original loop
    # Loop: for i = 1; i < n_elements; i += 2
    # Valid indices: 1, 3, 5, ..., up to n_elements-1
    num_iterations = (n_elements - 1) // 2
    
    if num_iterations <= 0:
        return
    
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(num_iterations, BLOCK_SIZE)
    
    s111_kernel[grid_size,](
        a, b, n_elements, BLOCK_SIZE
    )