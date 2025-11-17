import torch
import triton
import triton.language as tl

@triton.jit
def s1244_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid elements (excluding last element since we process n-1 elements)
    mask = offsets < (n_elements - 1)
    
    # Load inputs with masking
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Load a[i+1] values (OLD values before update)
    offsets_plus_1 = offsets + 1
    mask_plus_1 = offsets_plus_1 < n_elements
    a_plus_1_vals = tl.load(a_ptr + offsets_plus_1, mask=mask_plus_1)
    
    # Compute new a[i] = b[i] + c[i]*c[i] + b[i]*b[i] + c[i]
    c_squared = c_vals * c_vals
    b_squared = b_vals * b_vals
    new_a_vals = b_vals + c_squared + b_squared + c_vals
    
    # Compute d[i] = NEW a[i] + OLD a[i+1]
    d_vals = new_a_vals + a_plus_1_vals
    
    # Store results with masking
    tl.store(a_ptr + offsets, new_a_vals, mask=mask)
    tl.store(d_ptr + offsets, d_vals, mask=mask)

def s1244_triton(a, b, c, d):
    """
    Triton implementation of TSVC s1244
    
    Key optimizations:
    - Vectorized computation using Triton blocks
    - Coalesced memory access patterns
    - Single kernel launch processes all elements
    - Uses masking for edge case handling
    """
    # Ensure tensors are contiguous
    a = a.contiguous()
    b = b.contiguous() 
    c = c.contiguous()
    d = d.contiguous()
    
    n_elements = a.numel()
    
    # Handle edge case: need at least 2 elements
    if n_elements < 2:
        return a, d
    
    # Choose block size (power of 2 for efficiency)
    BLOCK_SIZE = 256
    
    # Calculate grid size (process n-1 elements)
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    # Launch kernel
    s1244_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, d