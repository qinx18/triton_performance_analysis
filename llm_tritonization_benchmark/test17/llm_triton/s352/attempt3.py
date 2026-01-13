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
    
    # Sum partial results on CPU
    result = torch.sum(partial_sums)
    return result.item()