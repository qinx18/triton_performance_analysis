import triton
import triton.language as tl
import torch

@triton.jit
def s452_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    # Convert index to real_t (i+1)
    i_plus_1 = (idx + 1).to(tl.float32)
    
    # a[i] = b[i] + c[i] * (real_t) (i+1)
    result = b_vals + c_vals * i_plus_1
    
    tl.store(a_ptr + idx, result, mask=mask)

def s452_triton(a, b, c):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s452_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a