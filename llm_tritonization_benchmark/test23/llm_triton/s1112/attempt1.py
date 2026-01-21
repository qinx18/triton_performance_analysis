import triton
import triton.language as tl
import torch

@triton.jit
def s1112_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, N, BLOCK_SIZE):
        # Reverse indexing: start from highest indices
        reverse_offsets = N - 1 - (block_start + offsets)
        mask = (block_start + offsets) < N
        
        # Load b values
        b_vals = tl.load(b_ptr + reverse_offsets, mask=mask)
        
        # Compute a[i] = b[i] + 1.0
        a_vals = b_vals + 1.0
        
        # Store to a
        tl.store(a_ptr + reverse_offsets, a_vals, mask=mask)

def s1112_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)
    s1112_kernel[grid](a, b, N, BLOCK_SIZE=BLOCK_SIZE)