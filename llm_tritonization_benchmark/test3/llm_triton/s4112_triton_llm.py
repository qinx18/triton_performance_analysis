import torch
import triton
import triton.language as tl

@triton.jit
def s4112_kernel(
    a_ptr, b_ptr, ip_ptr,
    n_elements, s,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s4112 - indirect addressing with scaling.
    Optimizations:
    - Coalesced memory access where possible
    - Block-level processing for better memory bandwidth utilization
    - Masked operations for handling arbitrary tensor sizes
    """
    # Get program ID and compute element indices for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for handling edge cases where block extends beyond array
    mask = offsets < n_elements
    
    # Load indices for indirect addressing
    indices = tl.load(ip_ptr + offsets, mask=mask, other=0)
    
    # Perform indirect load from b using gathered indices
    # Note: This creates gather pattern but is necessary for indirect addressing
    gathered_b = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Load current values from a
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Compute scaled update: a[i] += b[ip[i]] * s
    result = a_vals + gathered_b * s
    
    # Store result back to a
    tl.store(a_ptr + offsets, result, mask=mask)

def s4112_triton(a, b, ip, s):
    """
    Triton implementation of TSVC s4112 - indirect addressing with scaling.
    
    Key optimizations:
    - Block-based processing for better GPU utilization
    - Single kernel launch reduces overhead
    - Efficient memory access patterns within blocks
    """
    # Ensure tensors are contiguous for optimal memory access
    a = a.contiguous()
    b = b.contiguous()
    ip = ip.contiguous()
    
    n_elements = a.numel()
    
    # Choose block size for good occupancy - power of 2 for alignment
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s4112_kernel[grid](
        a, b, ip,
        n_elements, s,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a