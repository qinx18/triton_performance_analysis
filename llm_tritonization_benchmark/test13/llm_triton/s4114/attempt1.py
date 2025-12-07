import triton
import triton.language as tl
import torch

@triton.jit
def s4114_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, n1, n_elements, c_size, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Adjust indices to start from n1-1
    actual_indices = indices + (n1 - 1)
    
    # Mask for valid indices
    mask = (indices < n_elements) & (actual_indices >= 0)
    
    # Load ip values
    ip_vals = tl.load(ip_ptr + actual_indices, mask=mask, other=0)
    
    # Load b and d values
    b_vals = tl.load(b_ptr + actual_indices, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + actual_indices, mask=mask, other=0.0)
    
    # Calculate c indices: LEN_1D - k + 1 - 2 = c_size - ip_vals - 1
    c_indices = c_size - ip_vals - 1
    
    # Ensure c_indices are within bounds
    c_mask = mask & (c_indices >= 0) & (c_indices < c_size)
    
    # Load c values
    c_vals = tl.load(c_ptr + c_indices, mask=c_mask, other=0.0)
    
    # Compute result
    result = b_vals + c_vals * d_vals
    
    # Store result
    tl.store(a_ptr + actual_indices, result, mask=mask)

def s4114_triton(a, b, c, d, ip, n1):
    n_elements = a.shape[0] - (n1 - 1)
    c_size = c.shape[0]
    
    if n_elements <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s4114_kernel[grid](
        a, b, c, d, ip,
        n1, n_elements, c_size,
        BLOCK_SIZE=BLOCK_SIZE
    )