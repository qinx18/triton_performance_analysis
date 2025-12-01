import torch
import triton
import triton.language as tl

@triton.jit
def s221_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s221 computation with sequential dependency handling.
    Uses a single thread to maintain correct sequential order for b[i] dependency.
    """
    pid = tl.program_id(axis=0)
    
    # Only use the first program to handle sequential computation
    if pid != 0:
        return
    
    # Process elements sequentially to handle b[i-1] dependency
    for i in range(1, n_elements):
        # Load current elements
        a_val = tl.load(a_ptr + i)
        b_prev = tl.load(b_ptr + i - 1)  # Sequential dependency
        b_curr = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        # Compute updates
        new_a = a_val + c_val * d_val
        new_b = b_prev + new_a + d_val
        
        # Store results
        tl.store(a_ptr + i, new_a)
        tl.store(b_ptr + i, new_b)

def s221_triton(a, b, c, d):
    """
    Triton implementation of TSVC s221 function.
    Handles sequential dependency in b array computation.
    """
    # Ensure tensors are contiguous and on GPU
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    n_elements = a.numel()
    
    # Use single block since we need sequential processing
    # The kernel will handle the sequential loop internally
    grid = (1,)
    BLOCK_SIZE = 1024
    
    # Launch kernel
    s221_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b