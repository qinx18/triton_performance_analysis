import triton
import triton.language as tl
import torch

@triton.jit
def s254_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get block of indices
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load b values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # For i=0: x = b[N-1], for i>0: x = b[i-1]
    # Create shifted indices for previous values
    prev_offsets = offsets - 1
    prev_mask = offsets > 0
    
    # Load previous b values (b[i-1])
    b_prev = tl.load(b_ptr + prev_offsets, mask=prev_mask)
    
    # Load b[N-1] for i=0 case
    last_val = tl.load(b_ptr + (N - 1))
    
    # Select x value: b[N-1] for i=0, b[i-1] for i>0
    x_vals = tl.where(offsets == 0, last_val, b_prev)
    
    # Compute a[i] = (b[i] + x) * 0.5
    result = (b_vals + x_vals) * 0.5
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s254_triton(a, b):
    N = a.shape[0]
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s254_kernel[grid](
        a, b, N,
        BLOCK_SIZE=BLOCK_SIZE
    )