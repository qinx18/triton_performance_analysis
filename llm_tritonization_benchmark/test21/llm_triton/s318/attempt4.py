import torch
import triton
import triton.language as tl

@triton.jit
def s318_kernel(a_ptr, result_ptr, inc, N, BLOCK_SIZE: tl.constexpr):
    # This is a global reduction problem - finding max absolute value and its index
    # We'll use a single thread block to handle the sequential nature
    
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    # Initialize with first element
    k = 0
    index = 0
    first_val = tl.load(a_ptr + k)
    max_val = tl.where(first_val < 0, -first_val, first_val)
    k += inc
    
    # Sequential scan through array with stride
    for i in range(1, N):
        # Bounds check without break
        current_val = tl.load(a_ptr + k)
        abs_val = tl.where(current_val < 0, -current_val, current_val)
        
        # Update max and index if we found a larger absolute value
        # This implements the inverse of the C condition (ABS(a[k]) <= max)
        if abs_val > max_val:
            max_val = abs_val
            index = i
            
        k += inc
    
    # Store result: max + index + 1 (as per C return statement)
    result = max_val + index + 1
    tl.store(result_ptr, result)

def s318_triton(a, abs, inc):
    N = a.shape[0]
    
    # Create output tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single thread block since this is inherently sequential
    BLOCK_SIZE = 256
    grid = (1,)
    
    s318_kernel[grid](
        a, result, inc, N, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()