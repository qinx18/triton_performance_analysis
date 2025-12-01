import triton
import triton.language as tl
import torch

@triton.jit
def vsumr_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for bounds checking
    mask = offsets < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Compute partial sum
    partial_sum = tl.sum(a_vals)
    
    # Store partial sum
    tl.store(output_ptr + pid, partial_sum)

def vsumr_triton(a):
    n_elements = a.numel()
    
    # Choose block size
    BLOCK_SIZE = 1024
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(grid_size, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    vsumr_kernel[(grid_size,)](
        a, partial_sums, n_elements, BLOCK_SIZE
    )
    
    # Final reduction on CPU/GPU
    total_sum = torch.sum(partial_sums)
    
    return total_sum