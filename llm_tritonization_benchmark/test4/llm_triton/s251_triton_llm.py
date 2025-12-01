import torch
import triton
import triton.language as tl

@triton.jit
def s251_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s251: a[i] = (b[i] + c[i] * d[i])^2
    Uses coalesced memory access and vectorized operations.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data with masking for edge cases
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    
    # Compute s = b + c * d, then a = s * s
    s = b + c * d
    result = s * s
    
    # Store result with masking
    tl.store(a_ptr + offsets, result, mask=mask)

def s251_triton(a, b, c, d):
    """
    Triton implementation of TSVC s251.
    Computes a[i] = (b[i] + c[i] * d[i])^2 with GPU acceleration.
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    n_elements = a.numel()
    
    # Use power-of-2 block size for optimal memory coalescing
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s251_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a