import torch
import triton
import triton.language as tl

@triton.jit
def s3111_kernel(a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    partial_sum = 0.0
    
    for start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        condition_mask = a_vals > 0.0
        valid_mask = mask & condition_mask
        
        masked_vals = tl.where(valid_mask, a_vals, 0.0)
        partial_sum += tl.sum(masked_vals)
    
    # Store partial sum for this block
    tl.store(a_ptr + n_elements + pid, partial_sum)

def s3111_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create extended buffer for partial sums
    extended_a = torch.cat([a, torch.zeros(num_blocks, dtype=a.dtype, device=a.device)])
    
    grid = (num_blocks,)
    s3111_kernel[grid](
        extended_a, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Sum up partial results
    partial_sums = extended_a[n_elements:]
    result = torch.sum(partial_sums)
    
    return result.item()