import triton
import triton.language as tl
import torch

@triton.jit
def s471_kernel(
    x_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and calculate offsets once at start
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    # Create mask for boundary checking
    mask = idx < n_elements
    
    # Load values from memory
    b_vals = tl.load(b_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Compute: x[i] = b[i] + d[i] * d[i]
    x_vals = b_vals + d_vals * d_vals
    tl.store(x_ptr + idx, x_vals, mask=mask)
    
    # Compute: b[i] = c[i] + d[i] * e[i]
    b_new = c_vals + d_vals * e_vals
    tl.store(b_ptr + idx, b_new, mask=mask)

def s471_triton(x, b, c, d, e):
    n_elements = x.numel()
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s471_kernel[grid](
        x, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )