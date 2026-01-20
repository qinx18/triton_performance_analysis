import triton
import triton.language as tl

@triton.jit
def s353_kernel(a_ptr, b_ptr, ip_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load indices for gather operation
    indices = tl.load(ip_ptr + offsets, mask=mask)
    
    # Gather from b array using indices
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Load a values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Compute saxpy: a[i] += alpha * b[ip[i]]
    result = a_vals + alpha * b_vals
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s353_triton(a, b, c, ip):
    n_elements = a.shape[0]
    alpha = c[0].item()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s353_kernel[grid](a, b, ip, alpha, n_elements, BLOCK_SIZE)