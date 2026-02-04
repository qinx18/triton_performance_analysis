import torch
import triton
import triton.language as tl

@triton.jit
def s221_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute a[i] += c[i] * d[i] in parallel
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Compute a[i] += c[i] * d[i]
    a_vals = a_vals + c_vals * d_vals
    
    # Store updated a values
    tl.store(a_ptr + indices, a_vals, mask=mask)

def s221_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # First compute a[i] += c[i] * d[i] in parallel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s221_kernel[grid](
        a, b, c, d, n_elements, BLOCK_SIZE
    )
    
    # Then compute b[i] = b[i-1] + a[i] + d[i] sequentially using prefix sum
    # This is a sequential dependency that needs special handling
    
    # Extract the addends: a[1:] + d[1:]
    addends = a[1:] + d[1:]
    
    # Compute prefix sum of addends
    prefix_sums = torch.cumsum(addends, dim=0)
    
    # Update b[1:] = b[0] + prefix_sums
    b[1:] = b[0] + prefix_sums