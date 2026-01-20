import triton
import triton.language as tl
import torch

@triton.jit
def s318_kernel(a_ptr, n, inc, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    # Initialize
    k = 0
    index = 0
    
    # Load first element
    first_val = tl.load(a_ptr + 0)
    max_val = tl.abs(first_val)
    k += inc
    
    # Sequential loop for strided access
    for i in range(1, n):
        if k >= n:
            break
        
        # Load current element
        curr_val = tl.load(a_ptr + k)
        abs_curr = tl.abs(curr_val)
        
        # Update if current is greater than max
        if abs_curr > max_val:
            index = i
            max_val = abs_curr
        
        k += inc
    
    # Store results
    tl.store(a_ptr + n, max_val)
    tl.store(a_ptr + n + 1, index)

def s318_triton(a, inc):
    n = a.shape[0]
    
    if n == 0:
        return torch.tensor(1.0, device=a.device, dtype=a.dtype)
    
    # Create extended array for output
    extended_a = torch.cat([a, torch.zeros(2, device=a.device, dtype=a.dtype)])
    
    # Launch kernel with single thread
    BLOCK_SIZE = 1
    grid = (1,)
    s318_kernel[grid](extended_a, n, inc, BLOCK_SIZE=BLOCK_SIZE)
    
    # Extract results
    max_val = extended_a[n]
    index = extended_a[n + 1]
    
    return max_val + index + 1