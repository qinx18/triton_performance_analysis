import triton
import triton.language as tl
import torch

@triton.jit
def s353_kernel(a_ptr, b_ptr, ip_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute base offset
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once - must be power of 2
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate actual indices for this block
    indices = block_start + offsets
    
    # Mask for valid elements
    mask = indices < n_elements
    
    # Load indirect indices
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Create mask for valid ip indices (assuming they're within bounds)
    ip_mask = mask
    
    # Load values from b using indirect addressing
    b_vals = tl.load(b_ptr + ip_vals, mask=ip_mask, other=0.0)
    
    # Load current values from a
    a_vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    
    # Perform saxpy operation: a[i] += alpha * b[ip[i]]
    result = a_vals + alpha * b_vals
    
    # Store back to a
    tl.store(a_ptr + indices, result, mask=mask)

def s353_triton(a, b, c, ip):
    n_elements = a.shape[0]
    alpha = c[0].item()
    
    # Use power of 2 block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    grid = (grid_size,)
    
    # Launch kernel
    s353_kernel[grid](
        a, b, ip, alpha, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )