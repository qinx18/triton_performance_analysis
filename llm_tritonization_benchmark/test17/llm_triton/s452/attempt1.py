import torch
import triton
import triton.language as tl

@triton.jit
def s452_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    # Convert idx to float and add 1 (equivalent to (real_t)(i+1))
    idx_plus_1 = (idx + 1).to(tl.float32)
    
    result = b_vals + c_vals * idx_plus_1
    
    tl.store(a_ptr + idx, result, mask=mask)

def s452_triton(a, b, c):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s452_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)