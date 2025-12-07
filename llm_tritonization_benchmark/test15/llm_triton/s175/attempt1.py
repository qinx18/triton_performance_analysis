import triton
import triton.language as tl
import torch

@triton.jit
def s175_kernel(a_ptr, a_copy_ptr, b_ptr, inc, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(0, n, inc):
        current_offset = block_start + offsets
        valid_idx = current_offset + i
        
        mask = (valid_idx < n - 1) & ((valid_idx % inc) == (i % inc))
        
        a_copy_vals = tl.load(a_copy_ptr + valid_idx + inc, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + valid_idx, mask=mask, other=0.0)
        
        result = a_copy_vals + b_vals
        
        tl.store(a_ptr + valid_idx, result, mask=mask)

def s175_triton(a, b, inc):
    n = a.shape[0]
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s175_kernel[grid](
        a, a_copy, b, inc, n,
        BLOCK_SIZE=BLOCK_SIZE
    )