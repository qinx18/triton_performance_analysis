import triton
import triton.language as tl
import torch

@triton.jit
def s311_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    partial_sum = tl.sum(a_vals)
    
    tl.store(output_ptr + pid, partial_sum)

def s311_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    partial_sums = torch.zeros(grid_size, dtype=a.dtype, device=a.device)
    
    s311_kernel[(grid_size,)](
        a, partial_sums, n_elements, BLOCK_SIZE
    )
    
    result = torch.sum(partial_sums)
    return result