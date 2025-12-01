import torch
import triton
import triton.language as tl

@triton.jit
def s274_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input data with masking
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # First compute temp_a = c[i] + e[i] * d[i]
    temp_a = c_vals + e_vals * d_vals
    
    # Apply conditional logic
    condition = temp_a > 0.0
    
    # Where condition is true: a[i] = temp_a[i], b[i] = temp_a[i] + b[i]
    # Where condition is false: a[i] = d[i] * e[i], b[i] remains unchanged
    a_result = tl.where(condition, temp_a, d_vals * e_vals)
    b_result = tl.where(condition, temp_a + b_vals, b_vals)
    
    # Store results with masking
    tl.store(a_ptr + offsets, a_result, mask=mask)
    tl.store(b_ptr + offsets, b_result, mask=mask)

def s274_triton(a, b, c, d, e):
    """
    Triton implementation of TSVC s274.
    Optimized with coalesced memory access and efficient branching.
    """
    # Ensure tensors are contiguous and on GPU
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    n_elements = a.numel()
    
    # Create output tensors
    a_result = torch.empty_like(a)
    b_result = torch.empty_like(b)
    
    # Copy input data to output tensors for in-place operations
    a_result.copy_(a)
    b_result.copy_(b)
    
    # Choose block size for optimal memory coalescing
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s274_kernel[grid](
        a_result, b_result, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a_result, b_result