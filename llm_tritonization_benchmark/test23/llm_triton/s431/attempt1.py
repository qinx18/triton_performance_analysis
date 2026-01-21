import triton
import triton.language as tl

@triton.jit
def s431_kernel(a_ptr, a_copy_ptr, b_ptr, k, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n
    
    a_copy_vals = tl.load(a_copy_ptr + indices + k, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    result = a_copy_vals + b_vals
    
    tl.store(a_ptr + indices, result, mask=mask)

def s431_triton(a, b, k):
    n = a.shape[0]
    
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s431_kernel[grid](a, a_copy, b, k, n, BLOCK_SIZE=BLOCK_SIZE)