import torch
import triton
import triton.language as tl

@triton.jit
def s4114_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, 
                 n_elements, n1, LEN_1D,
                 BLOCK_SIZE: tl.constexpr):
    
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Adjust indices to start from n1-1
    actual_indices = indices + (n1 - 1)
    mask = (indices < n_elements) & (actual_indices < LEN_1D)
    
    # Load ip values using actual indices
    ip_vals = tl.load(ip_ptr + actual_indices, mask=mask)
    
    # Load b[i] and d[i]
    b_vals = tl.load(b_ptr + actual_indices, mask=mask)
    d_vals = tl.load(d_ptr + actual_indices, mask=mask)
    
    # Calculate c array indices: LEN_1D - k + 1 - 2 = LEN_1D - k - 1
    c_indices = LEN_1D - ip_vals - 1
    
    # Load c values with bounds checking
    c_mask = mask & (c_indices >= 0) & (c_indices < LEN_1D)
    c_vals = tl.load(c_ptr + c_indices, mask=c_mask)
    
    # Compute a[i] = b[i] + c[LEN_1D-k+1-2] * d[i]
    result = b_vals + c_vals * d_vals
    
    # Store results
    tl.store(a_ptr + actual_indices, result, mask=mask)

def s4114_triton(a, b, c, d, ip, n1):
    LEN_1D = a.shape[0]
    n_elements = LEN_1D - (n1 - 1)
    
    if n_elements <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s4114_kernel[grid](
        a, b, c, d, ip,
        n_elements, n1, LEN_1D,
        BLOCK_SIZE=BLOCK_SIZE
    )