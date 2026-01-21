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
    
    # Load values and compute absolute values
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    abs_vals = tl.abs(vals)
    
    # Find maximum absolute value in this block
    block_max = tl.max(abs_vals)
    
    # Store result
    tl.store(output_ptr + pid, block_max)

def s3113_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Output tensor for partial results
    partial_results = torch.zeros(n_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (n_blocks,)
    s3113_kernel[grid](a, partial_results, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Find maximum among all partial results and a[0]
    result = torch.max(partial_results)
    first_abs = torch.abs(a[0])
    final_max = torch.max(result, first_abs).item()
    return final_max