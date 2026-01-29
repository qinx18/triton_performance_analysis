import triton
import triton.language as tl
import torch

@triton.jit
def s173_kernel(a_ptr, b_ptr, n_elements, k, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    result = a_vals + b_vals
    
    store_indices = indices + k
    store_mask = (indices < n_elements) & (store_indices < n_elements + k)
    
    tl.store(a_ptr + store_indices, result, mask=store_mask)

def s173_triton(a, b):
    N = a.shape[0]
    k = N // 2
    n_elements = N // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s173_kernel[grid](a, b, n_elements, k, BLOCK_SIZE)