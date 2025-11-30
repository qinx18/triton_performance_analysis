import torch
import triton
import triton.language as tl

@triton.jit
def vsumr_kernel(
    a_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load elements from array a
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Sum the elements in this block
    block_sum = tl.sum(a_vals)
    
    # Store the partial sum for this block
    tl.store(output_ptr + pid, block_sum)

def vsumr_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(n_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (n_blocks,)
    vsumr_kernel[grid](
        a, partial_sums, n_elements, BLOCK_SIZE
    )
    
    # Sum all partial sums to get final result
    total_sum = torch.sum(partial_sums)
    
    return total_sum.item()