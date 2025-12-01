import torch
import triton
import triton.language as tl

@triton.jit
def vpvts_kernel(
    a_ptr,
    b_ptr,
    s,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate starting position for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary handling
    mask = offsets < n_elements
    
    # Load data with masking for edge cases
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Vectorized computation: a[i] += b[i] * s
    result = a_vals + b_vals * s
    
    # Store result back to memory
    tl.store(a_ptr + offsets, result, mask=mask)

def vpvts_triton(a, b, s):
    """
    Triton implementation of TSVC vpvts function.
    Optimized for GPU with coalesced memory access and vectorized operations.
    """
    # Ensure tensors are contiguous for optimal memory access
    a = a.contiguous()
    b = b.contiguous()
    
    n_elements = a.numel()
    
    # Use power-of-2 block size for optimal GPU utilization
    BLOCK_SIZE = 1024
    
    # Calculate grid size with proper ceiling division
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with optimized block size
    vpvts_kernel[grid](
        a, b, s,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a