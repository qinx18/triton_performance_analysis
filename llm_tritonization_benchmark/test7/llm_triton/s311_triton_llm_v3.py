import torch
import triton
import triton.language as tl

@triton.jit
def s311_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    total_sum = 0.0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        block_sum = tl.sum(vals)
        total_sum += block_sum
    
    if tl.program_id(0) == 0:
        tl.store(output_ptr, total_sum)

def s311_triton(a):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    grid = (1,)
    s311_kernel[grid](a, output, n_elements, BLOCK_SIZE)
    
    return output