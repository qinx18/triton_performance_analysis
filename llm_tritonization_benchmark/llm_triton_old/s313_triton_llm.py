import torch
import triton
import triton.language as tl

@triton.jit
def s313_kernel(a_ptr, b_ptr, dot_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for dot product computation with block-level reduction
    """
    # Get block start position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset range for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data with masking for edge cases
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Compute element-wise multiplication and reduce within block
    products = a_vals * b_vals
    block_sum = tl.sum(products)
    
    # Atomic add to global dot product result
    tl.atomic_add(dot_ptr, block_sum)

def s313_triton(a, b):
    """
    Triton implementation of TSVC s313 - dot product computation
    Uses block-level reduction with atomic adds for global sum
    """
    a = a.contiguous()
    b = b.contiguous()
    
    n_elements = a.numel()
    
    # Initialize dot product result on GPU
    dot = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Choose block size for optimal memory coalescing
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with block-level reduction
    s313_kernel[grid](
        a, b, dot,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, b