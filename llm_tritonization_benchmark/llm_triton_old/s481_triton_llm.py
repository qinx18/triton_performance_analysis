import torch
import triton
import triton.language as tl

@triton.jit
def s481_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s481: a[i] += b[i] * c[i] with early exit check on d[i] < 0
    """
    # Get program ID and calculate offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load data with masking
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Check for negative values in d (early exit condition)
    # If any element is negative, we would exit in original code
    # Here we check if all valid elements are non-negative
    valid_elements = mask & (d_vals >= 0.0)
    
    # Only proceed if all elements in this block are valid (non-negative)
    # Load other arrays
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Compute a[i] += b[i] * c[i]
    result = a_vals + b_vals * c_vals
    
    # Store result back to memory
    tl.store(a_ptr + offsets, result, mask=mask)

def s481_triton(a, b, c, d):
    """
    Triton implementation of TSVC s481.
    
    Optimizations:
    - Vectorized memory access with configurable block size
    - Coalesced memory reads/writes for better bandwidth utilization
    - Efficient masking for handling array boundaries
    """
    # Ensure tensors are contiguous and on GPU
    a = a.contiguous()
    b = b.contiguous() 
    c = c.contiguous()
    d = d.contiguous()
    
    n_elements = a.numel()
    
    # Check if any element in d is negative (equivalent to exit condition)
    if torch.any(d < 0.0):
        # Return original array unchanged (equivalent to early exit)
        return a
    
    # Choose block size for optimal memory access patterns
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    s481_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a