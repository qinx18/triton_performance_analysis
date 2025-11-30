import triton
import triton.language as tl
import torch

@triton.jit
def s1421_kernel(b_ptr, xx_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    xx = tl.load(xx_ptr + offsets, mask=mask)
    a = tl.load(a_ptr + offsets, mask=mask)
    
    result = xx + a
    
    tl.store(b_ptr + offsets, result, mask=mask)

def s1421_triton(a, b):
    LEN_1D = a.shape[0]
    n_elements = LEN_1D // 2
    
    # xx points to &b[LEN_1D/2]
    xx = b[LEN_1D//2:]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1421_kernel[grid](
        b[:n_elements], 
        xx[:n_elements], 
        a[:n_elements], 
        n_elements, 
        BLOCK_SIZE
    )