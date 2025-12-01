import torch
import triton
import triton.language as tl

@triton.jit
def s3111_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    sum_val = 0.0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        condition_mask = a_vals > 0.0
        
        masked_vals = tl.where(condition_mask, a_vals, 0.0)
        block_sum = tl.sum(masked_vals)
        sum_val += block_sum
    
    if tl.program_id(0) == 0:
        tl.store(output_ptr, sum_val)

def s3111_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    grid = (1,)
    s3111_kernel[grid](
        a, output, n_elements, BLOCK_SIZE
    )
    
    return output.item()