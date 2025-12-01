import torch
import triton
import triton.language as tl

@triton.jit
def s243_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s243 function.
    Each thread block processes BLOCK_SIZE elements sequentially.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load current elements
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Load a[i+1] values for the third computation
    # For the last element, we don't need a[i+1] since n = len(a) - 1
    offsets_plus1 = offsets + 1
    mask_plus1 = offsets_plus1 < (n_elements + 1)  # Original array size
    a_plus1_vals = tl.load(a_ptr + offsets_plus1, mask=mask_plus1)
    
    # First computation: a[i] = b[i] + c[i] * d[i]
    a_vals = b_vals + c_vals * d_vals
    
    # Second computation: b[i] = a[i] + d[i] * e[i]
    b_vals = a_vals + d_vals * e_vals
    
    # Third computation: a[i] = b[i] + a[i+1] * d[i]
    a_vals = b_vals + a_plus1_vals * d_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    tl.store(b_ptr + offsets, b_vals, mask=mask)

def s243_triton(a, b, c, d, e):
    """
    Triton implementation of TSVC s243 function.
    Optimized for GPU execution with coalesced memory access.
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    n = len(a) - 1
    
    if n <= 0:
        return a, b
    
    # Use power-of-2 block size for optimal memory coalescing
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Launch kernel with sequential processing to maintain data dependencies
    s243_kernel[grid](
        a, b, c, d, e,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b