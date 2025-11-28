import triton
import triton.language as tl
import torch

@triton.jit
def s4114_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, n1, LEN_1D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Adjust offsets to start from n1-1
    actual_offsets = offsets + (n1 - 1)
    mask = actual_offsets < LEN_1D
    
    # Load indices and data
    ip_vals = tl.load(ip_ptr + actual_offsets, mask=mask)
    b_vals = tl.load(b_ptr + actual_offsets, mask=mask)
    d_vals = tl.load(d_ptr + actual_offsets, mask=mask)
    
    # Compute c indices: c[LEN_1D-k+1-2] = c[LEN_1D-k-1]
    c_indices = LEN_1D - ip_vals - 1
    c_mask = mask & (c_indices >= 0) & (c_indices < LEN_1D)
    c_vals = tl.load(c_ptr + c_indices, mask=c_mask, other=0.0)
    
    # Compute result
    result = b_vals + c_vals * d_vals
    
    # Store result
    tl.store(a_ptr + actual_offsets, result, mask=mask)

def s4114_triton(a, b, c, d, ip, n1):
    LEN_1D = a.shape[0]
    num_elements = LEN_1D - (n1 - 1)
    
    if num_elements <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    s4114_kernel[grid](
        a, b, c, d, ip, n1, LEN_1D, BLOCK_SIZE
    )