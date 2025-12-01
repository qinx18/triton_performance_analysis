import torch
import triton
import triton.language as tl

@triton.jit
def s171_kernel(a_ptr, b_ptr, inc, LEN_1D, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for s171: a[i * inc] += b[i]
    Uses coalesced memory access for b, handles strided access for a
    """
    # Get program ID and calculate thread indices
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid indices
    mask = offsets < LEN_1D
    
    # Load b values with coalesced access
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Calculate strided offsets for a array
    a_offsets = offsets * inc
    
    # Load current a values, update, and store back
    a_vals = tl.load(a_ptr + a_offsets, mask=mask)
    a_vals += b_vals
    tl.store(a_ptr + a_offsets, a_vals, mask=mask)

def s171_triton(a, b, inc):
    """
    TSVC function s171 - Triton implementation
    
    Optimizations:
    - Vectorized operations using Triton blocks
    - Coalesced memory access for array b
    - Efficient handling of strided access pattern for array a
    """
    a = a.contiguous()
    b = b.contiguous()
    
    LEN_1D = b.size(0)
    
    # Choose block size based on problem size
    BLOCK_SIZE = min(1024, triton.next_power_of_2(LEN_1D))
    
    # Calculate grid size
    grid = (triton.cdiv(LEN_1D, BLOCK_SIZE),)
    
    # Launch kernel
    s171_kernel[grid](
        a, b, inc, LEN_1D, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a