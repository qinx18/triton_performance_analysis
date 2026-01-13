import torch
import triton
import triton.language as tl

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Get block start position
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Mask for valid indices (0 to n-2)
    mask = indices < (n - 1)
    
    # Load values
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + indices, mask=mask, other=0.0)
    a_next_vals = tl.load(a_ptr + indices + 1, mask=mask, other=0.0)
    
    # Statement 0: a[i] = b[i] + c[i] * d[i]
    a_vals = b_vals + c_vals * d_vals
    tl.store(a_ptr + indices, a_vals, mask=mask)
    
    # Statement 1: b[i] = c[i] + b[i]
    b_new_vals = c_vals + b_vals
    tl.store(b_ptr + indices, b_new_vals, mask=mask)
    
    # Statement 2: a[i+1] = b[i] + a[i+1] * d[i]
    # This only affects the last iteration (i = n-2) due to overwrite optimization
    last_iter_mask = indices == (n - 2)
    final_mask = mask & last_iter_mask
    a_next_new_vals = b_new_vals + a_next_vals * d_vals
    tl.store(a_ptr + indices + 1, a_next_new_vals, mask=final_mask)

def s244_triton(a, b, c, d):
    n = a.shape[0]
    
    # Block size
    BLOCK_SIZE = 256
    
    # Number of blocks needed for n-1 elements
    num_blocks = triton.cdiv(n - 1, BLOCK_SIZE)
    
    # Launch kernel
    s244_kernel[(num_blocks,)](a, b, c, d, n, BLOCK_SIZE)