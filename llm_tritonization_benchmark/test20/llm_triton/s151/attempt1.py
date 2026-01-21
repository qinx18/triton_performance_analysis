import triton
import triton.language as tl

@triton.jit
def s151_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n - 1
    
    a_vals = tl.load(a_ptr + indices + 1, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    result = a_vals + b_vals
    
    tl.store(a_ptr + indices, result, mask=mask)

def s151_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
    
    s151_kernel[grid](a, b, n, BLOCK_SIZE=BLOCK_SIZE)