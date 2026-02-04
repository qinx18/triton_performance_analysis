import triton
import triton.language as tl
import torch

@triton.jit
def s4113_kernel(a, b, c, ip, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    
    # Load indices from ip array
    indices = tl.load(ip + current_offsets, mask=mask)
    
    # Gather from b array using indirect addressing
    b_values = tl.load(b + indices, mask=mask)
    
    # Direct load from c array
    c_values = tl.load(c + current_offsets, mask=mask)
    
    # Compute result
    result = b_values + c_values
    
    # Scatter to a array using indirect addressing
    tl.store(a + indices, result, mask=mask)

def s4113_triton(a, b, c, ip):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s4113_kernel[grid](a, b, c, ip, n_elements, BLOCK_SIZE)
    
    return a