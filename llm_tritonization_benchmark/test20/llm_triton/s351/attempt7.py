import triton
import triton.language as tl
import torch

@triton.jit
def s351_kernel(a_ptr, b_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)

    # Compute: a[i] += alpha * b[i]
    result = a_vals + alpha * b_vals

    # Store result back to a
    tl.store(a_ptr + offsets, result, mask=mask)

def s351_triton(a, b, c):
    N = a.shape[0]
    alpha = c[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s351_kernel[grid](a, b, alpha, N, BLOCK_SIZE=BLOCK_SIZE)