import torch
import triton
import triton.language as tl

@triton.jit
def s291_expand_im1_kernel(im1_expanded_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    im1_val = n - 1
    for i in range(n):
        tl.store(im1_expanded_ptr + i, im1_val)
        im1_val = i

@triton.jit
def s291_kernel(a_ptr, b_ptr, im1_expanded_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    im1_vals = tl.load(im1_expanded_ptr + offsets, mask=mask)
    b_im1_vals = tl.load(b_ptr + im1_vals, mask=mask)
    
    result = (b_vals + b_im1_vals) * 0.5
    tl.store(a_ptr + offsets, result, mask=mask)

def s291_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    im1_expanded = torch.zeros(n, dtype=torch.int32, device=a.device)
    
    grid_expand = (1,)
    s291_expand_im1_kernel[grid_expand](im1_expanded, n, BLOCK_SIZE=BLOCK_SIZE)
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s291_kernel[grid](a, b, im1_expanded, n, BLOCK_SIZE=BLOCK_SIZE)