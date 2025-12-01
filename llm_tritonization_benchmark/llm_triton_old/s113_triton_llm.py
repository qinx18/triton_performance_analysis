import torch
import triton
import triton.language as tl

@triton.jit
def s113_kernel(a_ptr, b_ptr, a0_val, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for s113: a[i] = a[0] + b[i] for i from 1 to n-1
    Uses broadcasting of a[0] value across all threads
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to handle boundary conditions and skip index 0
    mask = (offsets < n_elements) & (offsets > 0)
    
    # Load b values with masking
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] = a[0] + b[i] using broadcasted a[0]
    result = a0_val + b_vals
    
    # Store result back to a with masking
    tl.store(a_ptr + offsets, result, mask=mask)

def s113_triton(a, b):
    """
    Triton implementation of TSVC s113.
    Optimized with efficient memory coalescing and broadcasting.
    """
    a = a.contiguous()
    b = b.contiguous()
    
    n_elements = a.numel()
    
    # Early return for edge case
    if n_elements <= 1:
        return a
    
    # Extract a[0] value once to avoid repeated memory access
    a0_val = a[0].item()
    
    # Choose block size for optimal memory throughput
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with broadcasted a[0] value
    s113_kernel[grid](
        a, b, a0_val, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a