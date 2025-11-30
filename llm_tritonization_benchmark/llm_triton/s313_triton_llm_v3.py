import torch
import triton
import triton.language as tl

@triton.jit
def s313_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    partial_dot = a_vals * b_vals
    block_sum = tl.sum(partial_dot)
    
    tl.store(output_ptr + pid, block_sum)

def s313_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    partial_sums = torch.zeros(grid_size, dtype=a.dtype, device=a.device)
    
    s313_kernel[(grid_size,)](
        a, b, partial_sums,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    dot = partial_sums.sum()
    return dot