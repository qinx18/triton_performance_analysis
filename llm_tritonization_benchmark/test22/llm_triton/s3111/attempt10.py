import triton
import triton.language as tl
import torch

@triton.jit
def s3111_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load values
    vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    
    # Apply condition: if a[i] > 0.0
    condition_mask = vals > 0.0
    
    # Only include values that meet both conditions
    filtered_vals = tl.where(condition_mask & mask, vals, 0.0)
    
    # Sum within this block
    block_sum = tl.sum(filtered_vals)
    
    # Atomically add to global result
    tl.atomic_add(result_ptr, block_sum)

def s3111_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Initialize result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s3111_kernel[grid](a, result, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return result.item()