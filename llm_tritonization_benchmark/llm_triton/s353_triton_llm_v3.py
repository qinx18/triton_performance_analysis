import torch
import triton
import triton.language as tl

@triton.jit
def s353_kernel(a_ptr, b_ptr, ip_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(5):
        current_offsets = block_start + offsets + i
        mask = current_offsets < n_elements
        
        # Load indices from ip array
        indices = tl.load(ip_ptr + current_offsets, mask=mask, other=0)
        
        # Load from b array using gathered indices
        b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
        
        # Load current a values
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute saxpy: a[i] += alpha * b[ip[i]]
        result = a_vals + alpha * b_vals
        
        # Store back to a
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s353_triton(a, b, c, ip):
    alpha = c[0].item()
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE * 5),)
    
    s353_kernel[grid](
        a, b, ip, alpha, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a