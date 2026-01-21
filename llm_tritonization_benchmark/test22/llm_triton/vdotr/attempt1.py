import triton
import triton.language as tl
import torch

@triton.jit
def vdotr_kernel(a_ptr, b_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Calculate the starting position for this block
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Create offset vector
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n
    
    # Load data with masking
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Compute element-wise product and sum
    block_dot = tl.sum(a_vals * b_vals, axis=0)
    
    # Store the partial sum for this block
    tl.store(output_ptr + tl.program_id(0), block_dot)

def vdotr_triton(a, b):
    N = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate number of blocks needed
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    vdotr_kernel[grid](a, b, partial_sums, N, BLOCK_SIZE=BLOCK_SIZE)
    
    # Sum all partial results
    dot = torch.sum(partial_sums)
    
    return dot.item()