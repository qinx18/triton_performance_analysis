import torch
import triton
import triton.language as tl

@triton.jit
def s4114_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, n1, LEN_1D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    loop_start = n1 - 1
    loop_size = LEN_1D - loop_start
    
    indices = block_start + offsets
    actual_indices = loop_start + indices
    
    mask = (indices < loop_size) & (actual_indices < LEN_1D)
    
    # Load ip values for indirect addressing
    ip_vals = tl.load(ip_ptr + actual_indices, mask=mask)
    
    # Load b and d values
    b_vals = tl.load(b_ptr + actual_indices, mask=mask)
    d_vals = tl.load(d_ptr + actual_indices, mask=mask)
    
    # Calculate c indices: LEN_1D - k + 1 - 2 = LEN_1D - k - 1
    c_indices = LEN_1D - ip_vals - 1
    c_vals = tl.load(c_ptr + c_indices, mask=mask)
    
    # Compute result
    result = b_vals + c_vals * d_vals
    
    # Store result
    tl.store(a_ptr + actual_indices, result, mask=mask)

def s4114_triton(a, b, c, d, ip, n1):
    LEN_1D = a.shape[0]
    loop_size = LEN_1D - (n1 - 1)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(loop_size, BLOCK_SIZE),)
    
    s4114_kernel[grid](a, b, c, d, ip, n1, LEN_1D, BLOCK_SIZE)
    
    return a