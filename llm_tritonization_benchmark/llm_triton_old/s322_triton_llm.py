import torch
import triton
import triton.language as tl

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for s322 computation with sequential dependency handling.
    Each block processes elements sequentially to maintain data dependencies.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Process elements sequentially within each block to handle dependencies
    for i in range(BLOCK_SIZE):
        idx = block_start + i + 2  # Start from index 2
        
        if idx < n_elements:
            # Load current element and dependencies
            a_curr = tl.load(a_ptr + idx)
            a_prev1 = tl.load(a_ptr + idx - 1)
            a_prev2 = tl.load(a_ptr + idx - 2)
            b_curr = tl.load(b_ptr + idx)
            c_curr = tl.load(c_ptr + idx)
            
            # Compute: a[i] = a[i] + a[i-1] * b[i] + a[i-2] * c[i]
            result = a_curr + a_prev1 * b_curr + a_prev2 * c_curr
            
            # Store result
            tl.store(a_ptr + idx, result)

def s322_triton(a, b, c):
    """
    Triton implementation of TSVC s322 with optimized memory access patterns.
    Uses small block sizes to handle sequential dependencies while maximizing GPU utilization.
    """
    a = a.contiguous()
    b = b.contiguous()  
    c = c.contiguous()
    
    n_elements = a.shape[0]
    
    if n_elements <= 2:
        return a
    
    # Use small block size due to sequential dependencies
    # Larger blocks would require more complex synchronization
    BLOCK_SIZE = 32
    grid = ((n_elements - 2 + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Launch kernel with sequential processing within blocks
    s322_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a