import torch
import triton
import triton.language as tl

@triton.jit
def s318_kernel(a_ptr, result_ptr, n, inc, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid > 0:
        return
    
    # Initialize with first element
    k = 0
    index = 0
    max_val = tl.load(a_ptr + k)
    max_val = tl.where(max_val >= 0, max_val, -max_val)
    k += inc
    
    # Sequential search for maximum absolute value
    for i in range(1, n):
        # Check bounds
        if k >= n * inc:
            k += inc
            continue
            
        val = tl.load(a_ptr + k)
        abs_val = tl.where(val >= 0, val, -val)
        
        # Update max and index if current value is larger
        if abs_val > max_val:
            index = i
            max_val = abs_val
            
        k += inc
    
    # Store results
    tl.store(result_ptr, max_val)
    tl.store(result_ptr + 1, index.to(tl.float32))

def s318_triton(a, abs, inc):
    n = a.shape[0]
    
    # Result tensor to store max value and index
    result = torch.zeros(2, dtype=a.dtype, device=a.device)
    
    grid = (1,)
    BLOCK_SIZE = 256
    
    s318_kernel[grid](
        a, result, n, inc, BLOCK_SIZE=BLOCK_SIZE
    )
    
    max_val = result[0].item()
    index = int(result[1].item())
    
    return max_val + index + 1