import triton
import triton.language as tl
import torch

@triton.jit
def s313_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate global offset for this block
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE) + block_start
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load elements with masking
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Compute dot product for this block
    block_dot = tl.sum(a_vals * b_vals, axis=0)
    
    # Store the partial sum
    tl.store(output_ptr + tl.program_id(0), block_dot)

def s313_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate number of blocks needed
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    s313_kernel[grid](a, b, partial_sums, N, BLOCK_SIZE=BLOCK_SIZE)
    
    # Sum all partial results to get final dot product
    dot = torch.sum(partial_sums).item()
    
    return dot