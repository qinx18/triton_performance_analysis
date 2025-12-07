import torch
import triton
import triton.language as tl

@triton.jit
def s3113_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Load first element for initial max
    first_val = tl.load(a_ptr)
    abs_first = tl.abs(first_val)
    current_max = abs_first
    
    # Process array in blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        abs_vals = tl.abs(vals)
        
        # Find maximum in this block
        block_max = tl.max(abs_vals, axis=0)
        
        # Update global maximum
        current_max = tl.maximum(current_max, block_max)
    
    # Store result (single thread writes the final result)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, current_max)

def s3113_triton(a, abs):
    n_elements = a.shape[0]
    
    # Create output tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single program
    BLOCK_SIZE = 1024
    
    s3113_kernel[(1,)](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()