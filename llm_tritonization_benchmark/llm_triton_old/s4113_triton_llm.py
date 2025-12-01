import torch
import triton
import triton.language as tl

@triton.jit
def s4113_kernel(
    a_ptr, b_ptr, c_ptr, ip_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid elements
    mask = offsets < n_elements
    
    # Load indices and direct values
    indices = tl.load(ip_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Load indirect values using gathered indices
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute result and store back to indirect locations
    result = b_vals + c_vals
    tl.store(a_ptr + indices, result, mask=mask)

def s4113_triton(a, b, c, ip):
    """
    Triton implementation of TSVC s4113 - indirect addressing.
    Optimized for GPU with coalesced memory access where possible.
    """
    # Ensure contiguous memory layout
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    ip = ip.contiguous()
    
    n_elements = c.numel()
    
    # Choose block size for optimal memory throughput
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with indirect addressing pattern
    s4113_kernel[grid](
        a, b, c, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a