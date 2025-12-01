import torch
import triton
import triton.language as tl

@triton.jit
def s4117_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute block start
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid elements
    mask = offsets < n_elements
    
    # Load b and d arrays
    b = tl.load(b_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    
    # Compute c indices (i // 2) and load c values
    c_offsets = offsets // 2
    c = tl.load(c_ptr + c_offsets, mask=mask)
    
    # Compute result: b[i] + c[i//2] * d[i]
    result = b + c * d
    
    # Store result to a
    tl.store(a_ptr + offsets, result, mask=mask)

def s4117_triton(a, b, c, d):
    """
    Triton implementation of TSVC s4117
    Optimized for coalesced memory access and efficient integer division
    """
    a = a.contiguous()
    b = b.contiguous() 
    c = c.contiguous()
    d = d.contiguous()
    
    n_elements = a.numel()
    
    # Choose block size - power of 2 for efficiency
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s4117_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a