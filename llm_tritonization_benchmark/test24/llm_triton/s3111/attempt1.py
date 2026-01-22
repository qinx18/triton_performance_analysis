import torch
import triton
import triton.language as tl

@triton.jit
def s3111_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=0.0)
    
    # Apply condition: if a[i] > 0
    condition_mask = a_vals > 0.0
    
    # Set values to 0 where condition is false
    filtered_vals = tl.where(condition_mask, a_vals, 0.0)
    
    # Sum the filtered values in this block
    block_sum = tl.sum(filtered_vals)
    
    # Store the block sum
    tl.store(result_ptr + pid, block_sum)

def s3111_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for block sums
    block_sums = torch.zeros(n_blocks, dtype=a.dtype, device=a.device)
    
    grid = (n_blocks,)
    s3111_kernel[grid](a, block_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Sum all block results to get final sum
    total_sum = torch.sum(block_sums)
    
    return total_sum.item()