import torch
import triton
import triton.language as tl

@triton.jit
def s321_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Each thread block processes a range of elements
    for i in range(1, n_elements):
        # Check if this thread block should handle element i
        if i >= block_start and i < block_start + BLOCK_SIZE:
            # Load a[i-1], a[i], and b[i]
            prev_val = tl.load(a_ptr + (i-1))
            curr_val = tl.load(a_ptr + i)
            b_val = tl.load(b_ptr + i)
            
            # Compute a[i] += a[i-1] * b[i]
            new_val = curr_val + prev_val * b_val
            
            # Store result
            tl.store(a_ptr + i, new_val)

def s321_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s321_kernel[grid](
        a, b,
        n_elements,
        BLOCK_SIZE
    )
    
    return a