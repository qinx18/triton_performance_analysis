import torch
import triton
import triton.language as tl

@triton.jit
def s319_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s319: vectorized element-wise operations
    a[i] = c[i] + d[i]
    b[i] = c[i] + e[i]
    """
    # Get program ID and compute offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary handling
    mask = offsets < n_elements
    
    # Load input vectors with masking
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    e = tl.load(e_ptr + offsets, mask=mask)
    
    # Compute results: reuse c to minimize memory operations
    a_result = c + d
    b_result = c + e
    
    # Store results with masking
    tl.store(a_ptr + offsets, a_result, mask=mask)
    tl.store(b_ptr + offsets, b_result, mask=mask)

def s319_triton(a, b, c, d, e):
    """
    Triton implementation of TSVC s319 function.
    Performs vectorized element-wise additions:
    a[i] = c[i] + d[i]
    b[i] = c[i] + e[i]
    """
    # Ensure tensors are contiguous for optimal memory access
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    n_elements = a.numel()
    
    # Choose block size for good occupancy and memory coalescing
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with flattened tensor views
    s319_kernel[grid](
        a.view(-1), b.view(-1), c.view(-1), d.view(-1), e.view(-1),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b