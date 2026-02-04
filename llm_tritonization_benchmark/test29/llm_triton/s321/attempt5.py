import torch
import triton
import triton.language as tl

@triton.jit
def s321_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    for i in range(1, n_elements):
        if block_start <= i < block_start + BLOCK_SIZE:
            # Load a[i] and b[i]
            a_i = tl.load(a_ptr + i)
            b_i = tl.load(b_ptr + i)
            # Load a[i-1]
            a_prev = tl.load(a_ptr + (i - 1))
            # Update: a[i] += a[i-1] * b[i]
            result = a_i + a_prev * b_i
            tl.store(a_ptr + i, result)

def s321_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1
    
    grid = (1,)
    s321_kernel[grid](
        a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )