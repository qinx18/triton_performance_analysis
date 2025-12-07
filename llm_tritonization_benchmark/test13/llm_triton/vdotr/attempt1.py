import torch
import triton
import triton.language as tl

@triton.jit
def vdotr_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
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
    b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
    
    # Compute dot product contribution
    products = a_vals * b_vals
    
    # Sum within block
    block_sum = tl.sum(products)
    
    # Store block sum
    tl.store(output_ptr + pid, block_sum)

def vdotr_triton(a, b):
    # Ensure inputs are contiguous
    a = a.contiguous()
    b = b.contiguous()
    
    n_elements = a.numel()
    
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Allocate output tensor for partial sums
    partial_sums = torch.zeros(grid_size, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    vdotr_kernel[(grid_size,)](
        a,
        b,
        partial_sums,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Sum all partial results
    dot = torch.sum(partial_sums)
    
    return dot