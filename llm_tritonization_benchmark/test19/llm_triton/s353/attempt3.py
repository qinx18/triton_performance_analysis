import triton
import triton.language as tl

@triton.jit
def s353_kernel(a_ptr, b_ptr, c_ptr, ip_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load alpha (scalar)
    alpha = tl.load(c_ptr)
    
    # Load current values of a
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Load indices
    indices = tl.load(ip_ptr + offsets, mask=mask, other=0)
    
    # Create mask for valid indices to prevent out-of-bounds access
    valid_mask = mask & (indices >= 0) & (indices < n)
    
    # Gather from b using indices
    b_vals = tl.load(b_ptr + indices, mask=valid_mask, other=0.0)
    
    # Compute: a[i] += alpha * b[ip[i]]
    result = a_vals + alpha * b_vals
    
    # Store result back to a
    tl.store(a_ptr + offsets, result, mask=mask)

def s353_triton(a, b, c, ip):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s353_kernel[grid](
        a, b, c, ip,
        n, BLOCK_SIZE=BLOCK_SIZE
    )