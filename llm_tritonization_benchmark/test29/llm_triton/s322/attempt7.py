import triton
import triton.language as tl
import torch

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + 2
    
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < N
    
    # Load current values
    a_i = tl.load(a_ptr + indices, mask=mask)
    b_i = tl.load(b_ptr + indices, mask=mask)
    c_i = tl.load(c_ptr + indices, mask=mask)
    
    # Load previous values
    a_prev = tl.load(a_ptr + indices - 1, mask=mask)
    a_prev2 = tl.load(a_ptr + indices - 2, mask=mask)
    
    # Compute: a[i] = a[i] + a[i-1] * b[i] + a[i-2] * c[i]
    result = a_i + a_prev * b_i + a_prev2 * c_i
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s322_triton(a, b, c):
    N = a.shape[0]
    
    if N <= 2:
        return a
    
    BLOCK_SIZE = 32
    grid_size = triton.cdiv(N - 2, BLOCK_SIZE)
    grid = (grid_size,)
    
    s322_kernel[grid](
        a, b, c, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a