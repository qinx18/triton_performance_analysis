import triton
import triton.language as tl
import torch

@triton.jit
def vif_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    b_vals = tl.load(b_ptr + indices, mask=mask)
    condition = b_vals > 0.0
    
    a_vals = tl.load(a_ptr + indices, mask=mask)
    result = tl.where(condition, b_vals, a_vals)
    
    tl.store(a_ptr + indices, result, mask=mask)

def vif_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    vif_kernel[grid](a, b, N, BLOCK_SIZE)