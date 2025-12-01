import torch
import triton
import triton.language as tl

@triton.jit
def s431_kernel(
    a_ptr, b_ptr, a_out_ptr,
    n, k,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s431: a[i] = a[i+k] + b[i]
    Uses separate input/output pointers to avoid read-after-write hazards
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Calculate valid length (elements that can be computed)
    valid_len = n - k
    
    # Mask for valid indices within the block
    mask = offsets < valid_len
    
    # Load a[i+k] and b[i] with masking
    a_shifted = tl.load(a_ptr + offsets + k, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Compute result
    result = a_shifted + b_vals
    
    # Store result back to a[i]
    tl.store(a_out_ptr + offsets, result, mask=mask)

def s431_triton(a, b, k):
    """
    Triton implementation of TSVC s431.
    Optimized for coalesced memory access and efficient vectorization.
    """
    a = a.contiguous()
    b = b.contiguous()
    
    n = a.size(0)
    
    # Handle edge cases
    if k < 0 or k >= n:
        return a
    
    valid_len = n - k
    if valid_len <= 0:
        return a
    
    # Create output tensor to avoid read-after-write issues
    a_out = a.clone()
    
    # Choose block size for optimal memory throughput
    BLOCK_SIZE = 256
    grid = (triton.cdiv(valid_len, BLOCK_SIZE),)
    
    # Launch kernel
    s431_kernel[grid](
        a, b, a_out,
        n, k,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a_out