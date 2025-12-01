import torch
import triton
import triton.language as tl

@triton.jit
def s278_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get block start position
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Define offsets once at kernel start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate actual indices for this block
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load arrays
    a = tl.load(a_ptr + idx, mask=mask)
    b = tl.load(b_ptr + idx, mask=mask)
    c = tl.load(c_ptr + idx, mask=mask)
    d = tl.load(d_ptr + idx, mask=mask)
    e = tl.load(e_ptr + idx, mask=mask)
    
    # Implement the if-else logic using tl.where
    condition = a > 0.0
    
    # When a[i] <= 0: b[i] = -b[i] + d[i] * e[i], c[i] unchanged
    # When a[i] > 0: c[i] = -c[i] + d[i] * e[i], b[i] unchanged
    new_b = tl.where(condition, b, -b + d * e)
    new_c = tl.where(condition, -c + d * e, c)
    
    # Final computation: a[i] = b[i] + c[i] * d[i]
    new_a = new_b + new_c * d
    
    # Store results
    tl.store(a_ptr + idx, new_a, mask=mask)
    tl.store(b_ptr + idx, new_b, mask=mask)
    tl.store(c_ptr + idx, new_c, mask=mask)

def s278_triton(a, b, c, d, e):
    n_elements = a.numel()
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s278_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )