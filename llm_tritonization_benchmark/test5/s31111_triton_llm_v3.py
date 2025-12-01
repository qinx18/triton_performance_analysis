import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define offsets once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize sum
    sum_val = 0.0
    
    # Load values at specific indices and accumulate
    indices = [0, 4, 8, 12, 16, 20, 24, 28]
    
    for idx in indices:
        if idx < n_elements:
            val = tl.load(a_ptr + idx)
            sum_val += val
    
    # Store result (only first thread writes the sum)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, sum_val)

def s31111_triton(a):
    n_elements = a.numel()
    BLOCK_SIZE = 128
    
    # Create output tensor for the sum
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block since we're just computing one sum
    grid = (1,)
    
    s31111_kernel[grid](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result