import torch
import triton
import triton.language as tl

@triton.jit
def s3111_kernel(a_ptr, sum_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    partial_sum = 0.0
    
    for start in range(0, n_elements, BLOCK_SIZE * tl.num_programs(0)):
        current_offsets = start + block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        condition_mask = vals > 0.0
        valid_vals = tl.where(condition_mask & mask, vals, 0.0)
        partial_sum += tl.sum(valid_vals)
    
    tl.atomic_add(sum_ptr, partial_sum)

def s3111_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    num_programs = min(triton.cdiv(n_elements, BLOCK_SIZE), 256)
    
    sum_tensor = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    grid = (num_programs,)
    s3111_kernel[grid](
        a, sum_tensor, n_elements, BLOCK_SIZE
    )
    
    return sum_tensor.item()