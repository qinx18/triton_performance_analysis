import torch
import triton
import triton.language as tl

@triton.jit
def vsumr_kernel(a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    # Create mask for boundary conditions
    mask = current_offsets < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    
    # Sum the values in this block
    block_sum = tl.sum(a_vals)
    
    # Store the partial sum (we'll sum these on CPU)
    partial_sum_ptr = a_ptr + n_elements + pid
    tl.store(partial_sum_ptr, block_sum)

def vsumr_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Calculate number of blocks needed
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create extended tensor to store partial sums
    extended_a = torch.zeros(n_elements + n_blocks, dtype=a.dtype, device=a.device)
    extended_a[:n_elements] = a
    
    # Launch kernel
    grid = (n_blocks,)
    vsumr_kernel[grid](
        extended_a,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum the partial results on CPU for final reduction
    partial_sums = extended_a[n_elements:n_elements + n_blocks]
    total_sum = torch.sum(partial_sums)
    
    return total_sum.item()