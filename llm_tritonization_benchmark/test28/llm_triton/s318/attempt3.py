import triton
import triton.language as tl
import torch

@triton.jit
def s318_kernel(a_ptr, abs_ptr, inc, n, BLOCK_SIZE: tl.constexpr):
    # Find max absolute value and its index with stride
    pid = tl.program_id(0)
    
    if pid > 0:
        return
    
    # Initialize with first element
    k = 0
    max_val = tl.load(abs_ptr + 0)
    max_idx = 0
    k += inc
    
    # Process remaining elements
    for i in range(1, n):
        if k < n:
            current_abs = tl.load(abs_ptr + k)
            if current_abs > max_val:
                max_val = current_abs
                max_idx = i
        k += inc
    
    # Store results at the beginning of abs array
    tl.store(abs_ptr + 0, max_val)
    tl.store(a_ptr + 0, max_idx.to(tl.float32))

def s318_triton(a, abs, inc):
    n = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Make working copies only if inputs are tensors
    if torch.is_tensor(a):
        a_work = a.clone()
    else:
        a_work = torch.tensor([0.0], device=abs.device, dtype=abs.dtype)
        
    if torch.is_tensor(abs):
        abs_work = abs.clone()
    else:
        abs_work = torch.abs(a).clone()
    
    # Launch kernel with single block
    grid = (1,)
    s318_kernel[grid](a_work, abs_work, inc, n, BLOCK_SIZE)
    
    # Get results
    max_val = abs_work[0].item()
    max_idx = int(a_work[0].item())
    
    return max_val + max_idx + 1