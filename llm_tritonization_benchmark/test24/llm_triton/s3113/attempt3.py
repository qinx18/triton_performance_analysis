import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load values
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    
    # Compute absolute values and find block maximum
    abs_vals = tl.abs(vals)
    block_max = tl.max(abs_vals)
    
    # Store block result
    tl.store(output_ptr + pid, block_max)

@triton.jit
def s3113_final_kernel(partial_results_ptr, output_ptr, first_element, num_blocks, BLOCK_SIZE: tl.constexpr):
    # Start with abs(a[0]) as initial max
    current_max = tl.abs(first_element)
    
    # Process all partial results in blocks
    for block_start in range(0, num_blocks, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE)
        current_offsets = block_start + offsets
        mask = current_offsets < num_blocks
        
        # Load partial results
        vals = tl.load(partial_results_ptr + current_offsets, mask=mask, other=0.0)
        
        # Update maximum
        block_max = tl.max(vals)
        current_max = tl.maximum(current_max, block_max)
    
    # Store final result
    tl.store(output_ptr, current_max)

def s3113_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # First pass: compute block maxima
    partial_results = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    grid = (num_blocks,)
    
    s3113_kernel[grid](
        a, partial_results, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Second pass: reduce with proper initialization to abs(a[0])
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    reduction_grid = (1,)
    
    s3113_final_kernel[reduction_grid](
        partial_results, output, a[0], num_blocks, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()