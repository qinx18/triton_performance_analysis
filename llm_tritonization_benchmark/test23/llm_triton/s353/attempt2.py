import triton
import triton.language as tl
import torch

@triton.jit
def s353_kernel(a_ptr, b_ptr, alpha, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load indices - make sure they are valid
    indices = tl.load(ip_ptr + offsets, mask=mask, other=0)
    
    # Load current values of a
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Create mask for valid indices to prevent out-of-bounds access
    valid_indices_mask = mask & (indices >= 0) & (indices < n_elements)
    
    # Gather from b using indices with bounds checking
    b_vals = tl.load(b_ptr + indices, mask=valid_indices_mask, other=0.0)
    
    # Compute saxpy: a[i] += alpha * b[ip[i]]
    result = tl.where(mask, a_vals + alpha * b_vals, a_vals)
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s353_triton(a, b, c, ip):
    n_elements = a.shape[0]
    alpha = c[0].item()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s353_kernel[grid](
        a, b, alpha, ip, n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )