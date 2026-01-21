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
    
    # Load indices and values
    indices = tl.load(ip_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute saxpy: a[i] += alpha * b[ip[i]]
    result = a_vals + alpha * b_vals
    
    # Store result
    tl.store(a_ptr + idx, result, mask=mask)

def s353_triton(a, b, c, ip):
    N = a.shape[0]
    alpha = c[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s353_kernel[grid](a, b, ip, alpha, N, BLOCK_SIZE=BLOCK_SIZE)