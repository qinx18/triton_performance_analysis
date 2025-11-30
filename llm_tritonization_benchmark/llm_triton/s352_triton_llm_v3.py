import torch
import triton
import triton.language as tl

@triton.jit
def s352_kernel(
    a_ptr, b_ptr, result_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one block of the dot product computation
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Initialize dot accumulator
    dot = 0.0
    
    # Process elements in blocks of 5 (unrolled)
    for i in range(0, n_elements, 5):
        # Check if we have at least 5 elements remaining
        if i + 4 < n_elements:
            # Load 5 consecutive elements from both arrays
            a_vals = tl.load(a_ptr + i + tl.arange(0, 5))
            b_vals = tl.load(b_ptr + i + tl.arange(0, 5))
            
            # Compute unrolled dot product for these 5 elements
            dot += tl.sum(a_vals * b_vals)
        else:
            # Handle remaining elements
            remaining = n_elements - i
            if remaining > 0:
                mask = tl.arange(0, 5) < remaining
                a_vals = tl.load(a_ptr + i + tl.arange(0, 5), mask=mask, other=0.0)
                b_vals = tl.load(b_ptr + i + tl.arange(0, 5), mask=mask, other=0.0)
                dot += tl.sum(a_vals * b_vals)
            break
    
    # Store the result for this block
    program_id = tl.program_id(0)
    tl.store(result_ptr + program_id, dot)

def s352_triton(a, b):
    n_elements = a.shape[0]
    
    # Use single block since we need to compute a single dot product
    BLOCK_SIZE = 1
    grid = (1,)
    
    # Create output tensor for the dot product result
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    s352_kernel[grid](
        a, b, result,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return result[0].item()