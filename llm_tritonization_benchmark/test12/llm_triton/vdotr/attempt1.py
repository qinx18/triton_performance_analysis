import torch
import triton
import triton.language as tl

@triton.jit
def vdotr_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    current_offsets = block_start + offsets
    
    # Create mask for valid elements
    mask = current_offsets < n_elements
    
    # Load elements
    a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
    
    # Compute element-wise product
    products = a_vals * b_vals
    
    # Sum within this block
    block_sum = tl.sum(products)
    
    # Store the block sum
    tl.store(output_ptr + pid, block_sum)

def vdotr_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    
    # Calculate number of blocks needed
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    vdotr_kernel[grid](
        a, b, partial_sums,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial results to get final dot product
    dot = torch.sum(partial_sums)
    return dot.item()