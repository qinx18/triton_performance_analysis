import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load values and compute absolute values
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=-float('inf'))
    abs_vals = tl.abs(vals)
    
    # Find maximum absolute value in this block
    block_max = tl.max(abs_vals)
    
    # Store the result
    tl.store(output_ptr + pid, block_max)

def s3113_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for partial results
    partial_results = torch.zeros(grid_size, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (grid_size,)
    s3113_kernel[grid](a, partial_results, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Final reduction on CPU/GPU
    final_max = torch.max(partial_results).item()
    
    return final_max