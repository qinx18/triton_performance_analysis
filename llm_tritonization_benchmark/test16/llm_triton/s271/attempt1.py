import triton
import triton.language as tl
import torch

@triton.jit
def s271_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    b_vals = tl.load(b_ptr + idx, mask=mask)
    condition = b_vals > 0.0
    valid_mask = mask & condition
    
    a_vals = tl.load(a_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=valid_mask)
    
    update = b_vals * c_vals
    result = tl.where(condition, a_vals + update, a_vals)
    
    tl.store(a_ptr + idx, result, mask=mask)

def s271_triton(a, b, c):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s271_kernel[grid](
        a, b, c, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )