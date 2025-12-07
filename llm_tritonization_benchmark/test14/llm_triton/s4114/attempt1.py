import triton
import triton.language as tl
import torch

@triton.jit
def s4114_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, n1, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Adjust indices to start from n1-1
    actual_indices = indices + (n1 - 1)
    
    # Mask for valid indices
    mask = (indices < n_elements) & (actual_indices < n_elements)
    
    # Load ip values
    k_vals = tl.load(ip_ptr + actual_indices, mask=mask, other=0)
    
    # Calculate c indices: LEN_1D - k + 1 - 2 = n_elements - k - 1
    c_indices = n_elements - k_vals - 1
    c_mask = mask & (c_indices >= 0) & (c_indices < n_elements)
    
    # Load input values
    b_vals = tl.load(b_ptr + actual_indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + c_indices, mask=c_mask, other=0.0)
    d_vals = tl.load(d_ptr + actual_indices, mask=mask, other=0.0)
    
    # Compute result
    result = b_vals + c_vals * d_vals
    
    # Store result
    tl.store(a_ptr + actual_indices, result, mask=mask)

def s4114_triton(a, b, c, d, ip, n1):
    n_elements = a.shape[0]
    actual_n_elements = n_elements - (n1 - 1)
    
    if actual_n_elements <= 0:
        return
    
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(actual_n_elements, BLOCK_SIZE)
    
    s4114_kernel[(grid_size,)](
        a, b, c, d, ip, 
        n1, n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )