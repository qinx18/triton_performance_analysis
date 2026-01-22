import triton
import triton.language as tl
import torch

@triton.jit
def s313_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate which block this program handles
    pid = tl.program_id(axis=0)
    
    # Calculate the starting offset for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle array boundaries
    mask = offsets < n_elements
    
    # Load values from arrays a and b
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Compute element-wise product and sum for this block
    products = a_vals * b_vals
    block_sum = tl.sum(products, axis=0)
    
    # Store the partial sum for this block
    tl.store(output_ptr + pid, block_sum)

def s313_triton(a, b):
    N = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(N, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(grid_size, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (grid_size,)
    s313_kernel[grid](a, b, partial_sums, N, BLOCK_SIZE=BLOCK_SIZE)
    
    # Sum all partial results to get final dot product
    dot = torch.sum(partial_sums)
    
    return dot.item()