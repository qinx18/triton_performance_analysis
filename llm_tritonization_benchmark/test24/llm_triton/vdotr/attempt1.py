import triton
import triton.language as tl
import torch

@triton.jit
def vdotr_kernel(a_ptr, b_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load data
    mask = offsets < n
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Compute dot product for this block
    block_dot = tl.sum(a_vals * b_vals, axis=0)
    
    # Store result
    tl.store(output_ptr, block_dot)

def vdotr_triton(a, b):
    n = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Calculate number of blocks needed
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    vdotr_kernel[grid](
        a, b, partial_sums,
        n, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial results
    dot = torch.sum(partial_sums)
    
    return dot.item()