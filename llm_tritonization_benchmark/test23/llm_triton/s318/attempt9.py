import triton
import triton.language as tl
import torch

@triton.jit
def s318_kernel(a_ptr, max_val_ptr, max_idx_ptr, n, inc, BLOCK_SIZE: tl.constexpr):
    # Each block processes one reduction independently
    block_id = tl.program_id(0)
    if block_id > 0:
        return
    
    # Initialize with first element
    first_val = tl.load(a_ptr)
    current_max = tl.abs(first_val)
    current_idx = 0
    
    # Process elements with stride
    k = inc
    for i in range(1, n):
        if k < n:
            val = tl.load(a_ptr + k)
            abs_val = tl.abs(val)
            if abs_val > current_max:
                current_max = abs_val
                current_idx = i
        k += inc
        if k >= n:
            break
    
    # Store results
    tl.store(max_val_ptr, current_max)
    tl.store(max_idx_ptr, current_idx)

def s318_triton(a, abs, inc):
    n = a.shape[0]
    
    # Allocate output tensors
    max_val = torch.zeros(1, dtype=a.dtype, device=a.device)
    max_idx = torch.zeros(1, dtype=torch.int32, device=a.device)
    
    # Launch kernel
    grid = (1,)
    s318_kernel[grid](
        a, max_val, max_idx, n, inc,
        BLOCK_SIZE=1
    )
    
    return max_val.item() + max_idx.item() + 1