import triton
import triton.language as tl
import torch

@triton.jit
def s252_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load arrays
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    
    # Compute s = b[i] * c[i]
    s_vals = b_vals * c_vals
    
    # For t, we need b[i-1] * c[i-1] for i > 0, and 0.0 for i = 0
    # Create shifted indices for i-1
    prev_indices = indices - 1
    prev_mask = (indices > 0) & mask
    
    # Load previous b and c values
    b_prev = tl.load(b_ptr + prev_indices, mask=prev_mask, other=0.0)
    c_prev = tl.load(c_ptr + prev_indices, mask=prev_mask, other=0.0)
    
    # Compute t = b[i-1] * c[i-1] for i > 0, 0.0 for i = 0
    t_vals = tl.where(indices > 0, b_prev * c_prev, 0.0)
    
    # Compute a[i] = s + t
    a_vals = s_vals + t_vals
    
    # Store results
    tl.store(a_ptr + indices, a_vals, mask=mask)

def s252_triton(a, b, c):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s252_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a