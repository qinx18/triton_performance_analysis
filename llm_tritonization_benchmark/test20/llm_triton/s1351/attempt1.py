import triton
import triton.language as tl

@triton.jit
def s1351_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    a_vals = b_vals + c_vals
    tl.store(a_ptr + offsets, a_vals, mask=mask)

def s1351_triton(a, b, c, __restrict__):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s1351_kernel[grid](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE)