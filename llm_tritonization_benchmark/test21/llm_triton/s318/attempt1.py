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
    max_val = tl.abs(first_val)
    k += inc
    
    # Sequential scan through array with stride
    for i in range(1, N):
        if k >= N * inc:  # Bounds check
            break
            
        current_val = tl.load(a_ptr + k)
        abs_val = tl.abs(current_val)
        
        # Update max and index if we found a larger absolute value
        if abs_val > max_val:
            max_val = abs_val
            index = i
            
        k += inc
    
    # Store result: max + index + 1 (as per C return statement)
    result = max_val + tl.cast(index, tl.float32) + 1.0
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