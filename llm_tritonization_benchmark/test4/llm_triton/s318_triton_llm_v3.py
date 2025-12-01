import torch
import triton
import triton.language as tl

@triton.jit
def s318_kernel(a_ptr, inc, n, max_val_ptr, index_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid > 0:
        return
    
    # Initialize with first element
    max_val = tl.abs(tl.load(a_ptr))
    index = 0
    
    # Sequential search through array with stride inc
    for i in range(1, n):
        k = i * inc
        val = tl.abs(tl.load(a_ptr + k))
        
        # Update max and index if current value is greater
        if val > max_val:
            max_val = val
            index = i
    
    # Store results
    tl.store(max_val_ptr, max_val)
    tl.store(index_ptr, index)

def s318_triton(a, inc_val=1):
    n = a.shape[0]
    
    # Output tensors
    max_val = torch.zeros(1, dtype=a.dtype, device=a.device)
    index = torch.zeros(1, dtype=torch.int32, device=a.device)
    
    # Launch kernel with single thread since this is sequential reduction
    grid = (1,)
    s318_kernel[grid](
        a, inc_val, n,
        max_val, index,
        BLOCK_SIZE=1
    )
    
    return max_val, index