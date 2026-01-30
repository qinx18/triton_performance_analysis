import triton
import triton.language as tl
import torch

@triton.jit
def vdotr_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and calculate block start
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    # Create mask for boundary checking
    mask = current_offsets < n_elements
    
    # Load data with masking
    a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
    
    # Compute dot product for this block
    block_dot = tl.sum(a_vals * b_vals, axis=0)
    
    # Store the partial result
    tl.store(output_ptr + pid, block_dot)

def vdotr_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate number of blocks needed
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for partial results
    partial_results = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    vdotr_kernel[grid](
        a, b, partial_results,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial results to get final dot product
    dot = torch.sum(partial_results)
    
    return dot.item()