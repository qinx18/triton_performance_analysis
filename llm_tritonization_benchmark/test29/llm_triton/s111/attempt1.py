import triton
import triton.language as tl
import torch

@triton.jit
def s111_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate starting offset for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate actual indices (original loop: i = 1; i < LEN_1D; i += 2)
    # Map thread indices to original loop indices
    indices = 1 + 2 * (block_start + offsets)
    
    # Mask for valid indices
    mask = (block_start + offsets) < n_elements
    mask = mask & (indices < n_elements)
    
    # Load a[i-1] and b[i]
    a_prev = tl.load(a_ptr + indices - 1, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute a[i] = a[i-1] + b[i]
    result = a_prev + b_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s111_triton(a, b):
    N = a.shape[0]
    
    # Calculate number of iterations in original loop
    # Original loop: for (int i = 1; i < LEN_1D; i += 2)
    # This generates indices: 1, 3, 5, ..., up to LEN_1D-1
    n_iterations = (N - 1 + 1) // 2  # Number of odd indices from 1 to N-1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_iterations, BLOCK_SIZE),)
    
    s111_kernel[grid](
        a, b, n_iterations, BLOCK_SIZE
    )
    
    return a