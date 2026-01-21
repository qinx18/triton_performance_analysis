import triton
import triton.language as tl
import torch

@triton.jit
def s3111_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    
    # Apply condition: if a[i] > 0.0
    condition_mask = vals > 0.0
    filtered_vals = tl.where(condition_mask & mask, vals, 0.0)
    
    block_sum = tl.sum(filtered_vals, axis=0)
    tl.atomic_add(result_ptr, block_sum)

def s3111_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create output tensor for the result
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s3111_kernel[grid](a, result, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return result.item()