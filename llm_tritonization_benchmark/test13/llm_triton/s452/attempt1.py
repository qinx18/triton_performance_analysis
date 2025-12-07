import torch
import triton
import triton.language as tl

@triton.jit
def s452_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    
    # Convert indices to float and add 1 (matching (real_t)(i+1))
    i_plus_1 = (indices + 1).to(tl.float32)
    
    result = b_vals + c_vals * i_plus_1
    
    tl.store(a_ptr + indices, result, mask=mask)

def s452_triton(a, b, c):
    n_elements = a.numel()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s452_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )