import triton
import triton.language as tl

@triton.jit
def vpvtv_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < N
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    result = a_vals + b_vals * c_vals
    
    tl.store(a_ptr + offsets, result, mask=mask)

def vpvtv_triton(a, b, c):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    vpvtv_kernel[grid](
        a, b, c,
        N, BLOCK_SIZE
    )