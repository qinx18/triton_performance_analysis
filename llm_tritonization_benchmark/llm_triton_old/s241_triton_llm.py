import torch
import triton
import triton.language as tl

@triton.jit
def s241_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s241 computation.
    Processes elements in blocks, handling the dependency between a[i] and b[i].
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load current elements
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Load a[i+1] for b computation, handle boundary
    offsets_plus1 = offsets + 1
    mask_plus1 = offsets_plus1 < (n_elements + 1)  # Original array size
    a_plus1_vals = tl.load(a_ptr + offsets_plus1, mask=mask_plus1)
    
    # Compute a[i] = b[i] * c[i] * d[i]
    a_new = b_vals * c_vals * d_vals
    
    # Compute b[i] = a[i] * a[i+1] * d[i]
    b_new = a_new * a_plus1_vals * d_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_new, mask=mask)
    tl.store(b_ptr + offsets, b_new, mask=mask)

def s241_triton(a, b, c, d):
    """
    Triton implementation of TSVC s241 function.
    Optimized with block processing and efficient memory access patterns.
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    n = len(a) - 1
    
    if n <= 0:
        return a, b
    
    # Use block size optimized for memory coalescing
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Launch kernel
    s241_kernel[grid](
        a, b, c, d,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b