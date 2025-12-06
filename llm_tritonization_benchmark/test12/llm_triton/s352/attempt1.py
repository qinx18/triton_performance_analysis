import torch
import triton
import triton.language as tl

@triton.jit
def s352_kernel(a_ptr, b_ptr, dot_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    products = a_vals * b_vals
    block_sum = tl.sum(products)
    
    tl.atomic_add(dot_ptr, block_sum)

def s352_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    dot = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    s352_kernel[grid](
        a, b, dot, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return dot.item()