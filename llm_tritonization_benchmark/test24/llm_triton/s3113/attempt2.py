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
def s3113_reduction_kernel(partial_results_ptr, output_ptr, num_blocks, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks
    
    # Load partial results
    vals = tl.load(partial_results_ptr + offsets, mask=mask, other=0.0)
    
    # Find global maximum
    global_max = tl.max(vals)
    
    # Store final result (only first thread writes)
    if tl.program_id(0) == 0:
        tl.store(output_ptr, global_max)

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
    
    # Second pass: reduce block maxima to final result
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    reduction_grid = (1,)
    
    s3113_reduction_kernel[reduction_grid](
        partial_results, output, num_blocks, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()