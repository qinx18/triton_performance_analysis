import torch
import triton
import triton.language as tl

@triton.jit
def s1279_kernel(
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
    
    # Load input arrays with masking
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    e = tl.load(e_ptr + offsets, mask=mask)
    
    # Create condition mask: a[i] < 0 AND b[i] > a[i]
    condition = (a < 0.0) & (b > a)
    
    # Compute update: d[i] * e[i]
    update = d * e
    
    # Apply conditional update: c[i] += d[i] * e[i] where condition is true
    result = tl.where(condition, c + update, c)
    
    # Store result back to memory
    tl.store(c_ptr + offsets, result, mask=mask)

def s1279_triton(a, b, c, d, e):
    """
    Triton implementation of TSVC s1279
    Optimized GPU kernel with coalesced memory access and vectorized operations
    """
    # Ensure tensors are contiguous and on same device
    device = a.device
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    n_elements = a.numel()
    
    # Choose block size for optimal memory coalescing
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s1279_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return c