import triton
import triton.language as tl
import torch

@triton.jit
def s4121_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    # f(b[i], c[i]) = b[i] * c[i]
    result = a_vals + (b_vals * c_vals)
    
    tl.store(a_ptr + idx, result, mask=mask)

def s4121_triton(a, b, c):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s4121_kernel[grid](a, b, c, N, BLOCK_SIZE)