import torch
import triton
import triton.language as tl

@triton.jit
def s318_kernel(a_ptr, result_ptr, inc, n, BLOCK_SIZE: tl.constexpr):
    # Initialize with first element
    k = 0
    first_val = tl.load(a_ptr + k)
    max_val = tl.where(first_val < 0, -first_val, first_val)
    max_idx = 0
    
    # Process remaining elements
    for i in range(1, n):
        k = i * inc
        if k < n:
            val = tl.load(a_ptr + k)
            abs_val = tl.where(val < 0, -val, val)
            if abs_val > max_val:
                max_val = abs_val
                max_idx = i
    
    # Store results
    if tl.program_id(0) == 0:
        tl.store(result_ptr, max_val)
        tl.store(result_ptr + 1, max_idx)

def s318_triton(a, abs, inc):
    n = a.shape[0]
    
    # Create result tensor to store max_val and max_idx
    result = torch.zeros(2, dtype=torch.float32, device=a.device)
    
    # Launch kernel with single block since this is a global reduction
    BLOCK_SIZE = 256
    grid = (1,)
    
    s318_kernel[grid](
        a, result, inc, n, BLOCK_SIZE=BLOCK_SIZE
    )
    
    max_val = result[0].item()
    max_idx = int(result[1].item())
    
    # Return max + index + 1 as specified in C code
    return max_val + max_idx + 1