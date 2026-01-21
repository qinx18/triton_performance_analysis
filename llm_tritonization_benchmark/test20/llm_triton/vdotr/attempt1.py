import torch
import triton
import triton.language as tl

@triton.jit
def vdotr_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    block_dot = tl.sum(a_vals * b_vals, axis=0)
    tl.store(output_ptr, block_dot)

def vdotr_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 1024
    
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    partial_sums = torch.zeros(num_blocks, device=a.device, dtype=a.dtype)
    
    grid = (num_blocks,)
    vdotr_kernel[grid](a, b, partial_sums, n, BLOCK_SIZE=BLOCK_SIZE)
    
    dot = torch.sum(partial_sums).item()
    return dot