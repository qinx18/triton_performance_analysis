import triton
import triton.language as tl
import torch

@triton.jit
def s254_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, N, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < N
        
        # Load b values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        # Compute previous values for scalar expansion
        # x = b[i-1] for i > 0, x = b[N-1] for i = 0
        prev_offsets = tl.where(current_offsets == 0, N - 1, current_offsets - 1)
        prev_mask = mask
        x_vals = tl.load(b_ptr + prev_offsets, mask=prev_mask)
        
        # Compute a[i] = (b[i] + x) * 0.5
        result = (b_vals + x_vals) * 0.5
        
        # Store result
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s254_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s254_kernel[grid](
        a, b, N,
        BLOCK_SIZE=BLOCK_SIZE
    )