import torch
import triton
import triton.language as tl

@triton.jit
def s314_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize maximum with negative infinity
    running_max = float('-inf')
    
    # Process array in blocks to find maximum
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        block_max = tl.max(vals, axis=0)
        
        # Update running maximum
        if block_max > running_max:
            running_max = block_max
    
    # Store result (only first thread writes)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, running_max)

def s314_triton(a):
    n_elements = a.numel()
    
    # Create output tensor for result
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single program
    BLOCK_SIZE = 1024
    
    s314_kernel[(1,)](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()