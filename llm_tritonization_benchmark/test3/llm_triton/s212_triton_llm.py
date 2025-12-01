import torch
import triton
import triton.language as tl

@triton.jit
def s212_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s212: a[i] *= c[i]; b[i] += a[i + 1] * d[i]
    Processes elements in blocks with coalesced memory access
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data with masking for bounds checking
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # First operation: a[i] *= c[i]
    a_updated = a_vals * c_vals
    tl.store(a_ptr + offsets, a_updated, mask=mask)
    
    # Load a[i+1] for second operation, handle boundary case
    offsets_plus_1 = offsets + 1
    mask_plus_1 = offsets_plus_1 < (n_elements + 1)  # Allow reading one past for a[i+1]
    a_plus_1 = tl.load(a_ptr + offsets_plus_1, mask=mask_plus_1, other=0.0)
    
    # Second operation: b[i] += a[i + 1] * d[i]
    b_updated = b_vals + a_plus_1 * d_vals
    tl.store(b_ptr + offsets, b_updated, mask=mask)

def s212_triton(a, b, c, d):
    """
    Triton implementation of TSVC s212 with optimized memory access patterns
    """
    # Ensure contiguous memory layout for optimal GPU memory access
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    # Process n-1 elements as per original algorithm
    n_elements = a.shape[0] - 1
    
    if n_elements <= 0:
        return a, b
    
    # Use power-of-2 block size for optimal memory coalescing
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with memory-optimized parameters
    s212_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b