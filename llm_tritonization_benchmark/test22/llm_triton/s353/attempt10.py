import triton
import triton.language as tl
import torch

@triton.jit
def s353_kernel(a_ptr, b_ptr, ip_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load indices and ensure they are within bounds for b array access
    indices = tl.load(ip_ptr + idx, mask=mask, other=0)
    
    # Load a values
    a_vals = tl.load(a_ptr + idx, mask=mask, other=0.0)
    
    # Create mask for valid b array accesses
    b_mask = mask & (indices >= 0) & (indices < n_elements)
    
    # Load b values using gathered indices with proper masking
    b_vals = tl.load(b_ptr + indices, mask=b_mask, other=0.0)
    
    # Compute saxpy: a[i] += alpha * b[ip[i]]
    result = a_vals + alpha * b_vals
    
    # Store result back to a
    tl.store(a_ptr + idx, result, mask=mask)

def s353_triton(a, b, c, ip):
    N = a.shape[0]
    alpha = c[0].item()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s353_kernel[grid](a, b, ip, alpha, N, BLOCK_SIZE=BLOCK_SIZE)