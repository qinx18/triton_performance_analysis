import triton
import triton.language as tl

@triton.jit
def s353_kernel(a_ptr, b_ptr, c_ptr, ip_ptr, n_elements, alpha, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load indices with bounds checking
    indices = tl.load(ip_ptr + offsets, mask=mask, other=0)
    
    # Create mask for valid indices to prevent out-of-bounds access
    valid_indices_mask = mask & (indices >= 0) & (indices < n_elements)
    
    # Load a values
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Gather from b using indices with proper masking
    b_vals = tl.load(b_ptr + indices, mask=valid_indices_mask, other=0.0)
    
    # Compute saxpy: a += alpha * b
    result = a_vals + alpha * b_vals
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s353_triton(a, b, c, ip):
    n_elements = a.shape[0]
    alpha = c[0].item()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s353_kernel[grid](
        a, b, c, ip,
        n_elements, alpha,
        BLOCK_SIZE=BLOCK_SIZE
    )