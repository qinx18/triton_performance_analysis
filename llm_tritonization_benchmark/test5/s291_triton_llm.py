import torch
import triton
import triton.language as tl

@triton.jit
def s291_kernel(
    a_ptr,
    b_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s291 operation.
    Each element a[i] = (b[i] + b[im1]) * 0.5 where im1 is the previous index (wrapping to n-1 for i=0).
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load current b values
    b_curr = tl.load(b_ptr + offsets, mask=mask)
    
    # Calculate previous indices (im1)
    # For i=0, im1 should be n-1; for i>0, im1 = i-1
    prev_offsets = tl.where(offsets == 0, n - 1, offsets - 1)
    
    # Load previous b values
    b_prev = tl.load(b_ptr + prev_offsets, mask=mask)
    
    # Compute result: (b[i] + b[im1]) * 0.5
    result = (b_curr + b_prev) * 0.5
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s291_triton(a, b):
    """
    Triton implementation of TSVC s291.
    Optimized using parallel processing with proper handling of previous index dependencies.
    """
    a = a.contiguous()
    b = b.contiguous()
    
    n = b.numel()
    
    # Choose block size for optimal memory coalescing
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Launch kernel
    s291_kernel[grid](
        a, b, n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a