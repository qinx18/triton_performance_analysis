import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with first element's absolute value
    first_val = tl.load(a_ptr)
    block_max = tl.abs(first_val)
    
    # Process blocks
    for start_idx in range(0, n_elements, BLOCK_SIZE):
        current_offsets = start_idx + offsets
        mask = current_offsets < n_elements
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        abs_vals = tl.abs(vals)
        block_max = tl.maximum(block_max, tl.max(abs_vals))
    
    # Store result (only first thread writes)
    if pid == 0:
        tl.store(result_ptr, block_max)

def s3113_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block
    grid = (1,)
    s3113_kernel[grid](a, result, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return result.item()