import torch
import triton
import triton.language as tl

@triton.jit
def s4113_kernel(a_ptr, b_ptr, c_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get block indices
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Create mask for valid indices
    mask = indices < n_elements
    
    # Load indices from ip array
    ip_indices = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Create mask for valid ip indices (assuming they are within bounds)
    ip_mask = mask
    
    # Load values using gathered indices
    b_vals = tl.load(b_ptr + ip_indices, mask=ip_mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    
    # Compute result
    result = b_vals + c_vals
    
    # Store result using scattered indices
    tl.store(a_ptr + ip_indices, result, mask=ip_mask)

def s4113_triton(a, b, c, ip):
    n_elements = ip.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s4113_kernel[grid](
        a, b, c, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )