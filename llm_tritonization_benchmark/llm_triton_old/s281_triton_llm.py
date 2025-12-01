import torch
import triton
import triton.language as tl

@triton.jit
def s281_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s281: x = a[LEN_1D-i-1] + b[i] * c[i]; a[i] = x - 1.0; b[i] = x
    """
    # Get program ID and compute offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Compute reverse indices for a: LEN_1D-i-1
    reverse_offsets = n_elements - 1 - offsets
    reverse_mask = reverse_offsets >= 0
    combined_mask = mask & reverse_mask
    
    # Load data with masking
    a_rev = tl.load(a_ptr + reverse_offsets, mask=combined_mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    
    # Compute x = a[LEN_1D-i-1] + b[i] * c[i]
    x = a_rev + b_vals * c_vals
    
    # Store results: a[i] = x - 1.0, b[i] = x
    tl.store(a_ptr + offsets, x - 1.0, mask=mask)
    tl.store(b_ptr + offsets, x, mask=mask)

def s281_triton(a, b, c):
    """
    Triton implementation of TSVC s281
    
    Optimizations:
    - Coalesced memory access patterns
    - Single kernel launch instead of element-wise operations
    - Efficient block-based processing with masking
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    n_elements = a.numel()
    
    # Choose block size for optimal memory access
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s281_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b