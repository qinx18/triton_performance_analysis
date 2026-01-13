import triton
import triton.language as tl
import torch

@triton.jit
def s352_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load elements
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Compute element-wise products
    products = a_vals * b_vals
    
    # Sum across the block (reduction)
    result = tl.sum(products)
    
    # Store partial result
    tl.store(output_ptr + pid, result)

@triton.jit
def s352_reduction_kernel(partial_sums_ptr, output_ptr, n_partials, BLOCK_SIZE: tl.constexpr):
    # Simple reduction kernel to sum partial results
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_partials
    
    vals = tl.load(partial_sums_ptr + offsets, mask=mask, other=0.0)
    result = tl.sum(vals)
    
    # Store final result
    tl.store(output_ptr, result)

def s352_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create tensor to store partial sums
    partial_sums = torch.zeros(grid_size, dtype=a.dtype, device=a.device)
    
    # Launch first kernel to compute partial dot products
    s352_kernel[(grid_size,)](
        a, b, partial_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # If we have multiple blocks, we need to sum the partial results
    if grid_size > 1:
        # Create output tensor
        output = torch.zeros(1, dtype=a.dtype, device=a.device)
        # Launch reduction kernel
        s352_reduction_kernel[(1,)](
            partial_sums, output, grid_size, BLOCK_SIZE=min(grid_size, 256)
        )
        return output[0].item()
    else:
        return partial_sums[0].item()