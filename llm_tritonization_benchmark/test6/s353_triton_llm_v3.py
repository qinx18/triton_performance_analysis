import triton
import triton.language as tl
import torch

@triton.jit
def s353_kernel(a_ptr, b_ptr, ip_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for unroll_start in range(0, BLOCK_SIZE, 5):
        current_offsets = block_start + unroll_start + offsets[:5]
        mask = current_offsets < n_elements
        
        # Load indices
        ip_vals = tl.load(ip_ptr + current_offsets, mask=mask)
        
        # Load b values using gathered indices
        b_vals = tl.load(b_ptr + ip_vals, mask=mask)
        
        # Load a values
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        
        # Compute and store result
        result = a_vals + alpha * b_vals
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s353_triton(a, b, ip, c):
    n_elements = a.shape[0]
    alpha = c[0].item()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s353_kernel[grid](
        a, b, ip, alpha, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )