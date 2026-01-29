import triton
import triton.language as tl

@triton.jit
def s131_kernel(a_ptr, a_copy_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    mask = i_offsets < (n - 1)
    
    a_copy_vals = tl.load(a_copy_ptr + (i_offsets + 1), mask=mask)
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    
    result = a_copy_vals + b_vals
    
    tl.store(a_ptr + i_offsets, result, mask=mask)

def s131_triton(a, b):
    n = a.shape[0]
    
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
    
    s131_kernel[grid](
        a, a_copy, b, n, BLOCK_SIZE
    )