import torch
import triton
import triton.language as tl

@triton.jit
def s352_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get block start position
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Initialize dot product accumulator
    dot_sum = 0.0
    
    # Process elements in steps of 5
    for i in range(0, n_elements, 5):
        if i + 4 < n_elements:  # Ensure we have 5 elements available
            # Load 5 consecutive elements from both arrays
            a_vals = tl.load(a_ptr + i + tl.arange(0, 5))
            b_vals = tl.load(b_ptr + i + tl.arange(0, 5))
            
            # Compute dot product for these 5 elements
            products = a_vals * b_vals
            dot_sum += tl.sum(products)
    
    # Store the result (only first thread writes the final result)
    if tl.program_id(0) == 0:
        tl.store(output_ptr, dot_sum)

def s352_triton(a, b):
    n_elements = a.numel()
    
    # Create output tensor for the dot product result
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Use single block since we need to accumulate across all elements
    BLOCK_SIZE = 1024
    grid = (1,)
    
    # Launch kernel
    s352_kernel[grid](
        a, b, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output.item()