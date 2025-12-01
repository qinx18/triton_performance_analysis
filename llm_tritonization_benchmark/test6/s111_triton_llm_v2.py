import torch
import triton
import triton.language as tl

@triton.jit
def s111_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for s111 - processes odd indices with dependency on previous element
    Each thread handles one odd index: a[i] = a[i-1] + b[i] where i is odd
    """
    # Calculate odd index for this thread
    pid = tl.program_id(axis=0)
    odd_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    actual_idx = 2 * odd_idx + 1  # Convert to actual odd indices (1, 3, 5, ...)
    
    # Mask for valid indices
    mask = actual_idx < n_elements
    
    # Load a[i-1] (previous elements)
    prev_idx = actual_idx - 1
    a_prev = tl.load(a_ptr + prev_idx, mask=mask)
    
    # Load b[i] (current elements)
    b_curr = tl.load(b_ptr + actual_idx, mask=mask)
    
    # Compute a[i] = a[i-1] + b[i]
    result = a_prev + b_curr
    
    # Store back to a[i]
    tl.store(a_ptr + actual_idx, result, mask=mask)

def s111_triton(a, b, iterations):
    """
    Triton implementation of s111 - Conditional store
    Processes odd indices sequentially due to data dependency
    """
    a = a.contiguous()
    b = b.contiguous()
    
    n_elements = a.shape[0]
    
    # Calculate number of odd indices to process
    n_odd_elements = (n_elements - 1 + 1) // 2  # Number of odd indices from 1 to n_elements-1
    
    if n_odd_elements <= 0:
        return a
    
    # Use smaller block size due to memory access pattern
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n_odd_elements, BLOCK_SIZE)
    
    # Process iterations sequentially due to data dependency within each iteration
    for _ in range(2 * iterations):
        if grid_size > 0:
            s111_kernel[(grid_size,)](
                a, b, n_elements,
                BLOCK_SIZE=BLOCK_SIZE
            )
    
    return a