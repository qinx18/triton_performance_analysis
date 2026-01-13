import triton
import triton.language as tl
import torch

@triton.jit
def vdotr_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get block start position
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    # Create mask for bounds checking
    mask = current_offsets < n_elements
    
    # Load data with masking
    a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
    
    # Compute dot product for this block
    block_dot = tl.sum(a_vals * b_vals)
    
    # Store result for this block
    tl.store(output_ptr + tl.program_id(0), block_dot)

def vdotr_triton(a, b):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Calculate number of blocks needed
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(n_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (n_blocks,)
    vdotr_kernel[grid](
        a, b, partial_sums,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial results
    dot = torch.sum(partial_sums)
    
    return dot.item()