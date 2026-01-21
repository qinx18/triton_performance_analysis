import torch
import triton
import triton.language as tl

@triton.jit
def vdotr_kernel(a_ptr, b_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < n
    
    a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
    
    block_dot = tl.sum(a_vals * b_vals)
    tl.store(output_ptr + pid, block_dot)

def vdotr_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    grid_size = triton.cdiv(n, BLOCK_SIZE)
    partial_sums = torch.zeros(grid_size, dtype=a.dtype, device=a.device)
    
    grid = (grid_size,)
    vdotr_kernel[grid](a, b, partial_sums, n, BLOCK_SIZE=BLOCK_SIZE)
    
    dot = torch.sum(partial_sums)
    return dot.item()