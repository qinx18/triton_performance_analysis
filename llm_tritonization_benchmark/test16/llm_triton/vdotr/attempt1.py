import triton
import triton.language as tl
import torch

@triton.jit
def vdotr_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate block start position
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    # Load data with masking
    mask = current_offsets < n_elements
    a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
    
    # Compute dot product for this block
    block_dot = tl.sum(a_vals * b_vals, axis=0)
    
    # Store the partial sum
    tl.store(output_ptr + tl.program_id(0), block_dot)

def vdotr_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Calculate number of blocks needed
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(n_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    vdotr_kernel[(n_blocks,)](
        a, b, partial_sums, n_elements, BLOCK_SIZE
    )
    
    # Sum all partial results
    dot = torch.sum(partial_sums)
    
    return dot.item()