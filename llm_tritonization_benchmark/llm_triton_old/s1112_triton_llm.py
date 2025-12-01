import torch
import triton
import triton.language as tl

@triton.jit
def s1112_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for s1112 operation.
    Performs: a[i] = b[i] + 1.0 for all elements in reverse order
    """
    # Get program ID and calculate offset
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle edge cases
    mask = offsets < n_elements
    
    # Load b values with masking
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Perform computation: add 1.0 to each element
    result = b_vals + 1.0
    
    # Store results back to a with masking
    tl.store(a_ptr + offsets, result, mask=mask)

def s1112_triton(a, b, iterations):
    """
    Triton implementation of TSVC s1112 function.
    Optimized GPU version that processes entire array in parallel per iteration.
    """
    a = a.contiguous()
    b = b.contiguous()
    
    n_elements = a.numel()
    
    # Choose block size for optimal memory coalescing
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Execute iterations * 3 times as in original
    for _ in range(iterations * 3):
        s1112_kernel[grid](
            a, b, n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    return a