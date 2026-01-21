import torch
import triton
import triton.language as tl

@triton.jit
def s318_kernel(a_ptr, result_ptr, inc, n, BLOCK_SIZE: tl.constexpr):
    # Single thread reduction
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    # Initialize with first element
    k = 0
    max_val = tl.load(a_ptr + k)
    max_abs_val = tl.where(max_val >= 0, max_val, -max_val)
    max_index = 0
    
    # Process elements with stride
    k += inc
    for i in range(1, n):
        if k < n:
            val = tl.load(a_ptr + k)
            val_abs = tl.where(val >= 0, val, -val)
            # Update if absolute value is greater
            update = val_abs > max_abs_val
            max_index = tl.where(update, i, max_index)
            max_abs_val = tl.where(update, val_abs, max_abs_val)
        k += inc
    
    # Store results
    tl.store(result_ptr, max_abs_val)
    tl.store(result_ptr + 1, max_index)

def s318_triton(a, abs, inc):
    n = a.shape[0]
    
    # Create result tensor to store max_abs and index
    result = torch.zeros(2, dtype=torch.float32, device=a.device)
    
    # Launch kernel with single block
    grid = (1,)
    BLOCK_SIZE = 256
    
    s318_kernel[grid](
        a, result, inc, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    max_abs_val = result[0].item()
    max_index = int(result[1].item())
    
    return max_abs_val + max_index + 1