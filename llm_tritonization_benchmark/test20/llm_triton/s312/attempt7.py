import triton
import triton.language as tl
import torch

@triton.jit
def _prod_combine(a, b):
    return a * b

@triton.jit
def s312_kernel(a_ptr, partial_products, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    ptrs = a_ptr + block_start + offsets
    mask = block_start + offsets < n
    
    vals = tl.load(ptrs, mask=mask, other=1.0)
    
    # Manually implement reduction within block
    block_prod = 1.0
    for i in range(BLOCK_SIZE):
        if i < tl.minimum(BLOCK_SIZE, n - block_start):
            elem = tl.load(a_ptr + block_start + i)
            block_prod = block_prod * elem
    
    tl.store(partial_products + pid, block_prod)

def s312_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    partial_products = torch.ones(num_blocks, dtype=a.dtype, device=a.device)
    
    grid = (num_blocks,)
    s312_kernel[grid](a, partial_products, n, BLOCK_SIZE=BLOCK_SIZE)
    
    prod = torch.prod(partial_products).item()
    return prod