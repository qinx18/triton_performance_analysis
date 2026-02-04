import triton
import triton.language as tl
import torch

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Get the starting position for this block
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    # Create masks for valid indices
    # Main computation is for i from 1 to n-1
    valid_mask = (idx >= 1) & (idx < n)
    
    # Load arrays for current indices
    a_vals = tl.load(a_ptr + idx, mask=valid_mask, other=0.0)
    b_vals = tl.load(b_ptr + idx, mask=valid_mask, other=0.0)
    c_vals = tl.load(c_ptr + idx, mask=valid_mask, other=0.0)
    d_vals = tl.load(d_ptr + idx, mask=valid_mask, other=0.0)
    
    # Load c[i-1] values for indices >= 1
    c_prev_vals = tl.load(c_ptr + (idx - 1), mask=valid_mask, other=0.0)
    
    # Compute transformations:
    # t = a[i] + b[i];
    # a[i] = t + c[i-1];
    # t = c[i] * d[i];
    # c[i] = t;
    
    # First scalar expansion: t = a[i] + b[i]
    t1 = a_vals + b_vals
    new_a = t1 + c_prev_vals
    
    # Second scalar expansion: t = c[i] * d[i]
    t2 = c_vals * d_vals
    new_c = t2
    
    # Store results
    tl.store(a_ptr + idx, new_a, mask=valid_mask)
    tl.store(c_ptr + idx, new_c, mask=valid_mask)

def s261_triton(a, b, c, d):
    n = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(n, BLOCK_SIZE)
    
    # Launch kernel
    s261_kernel[(grid_size,)](
        a, b, c, d, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, c