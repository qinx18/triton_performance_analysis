import torch
import triton
import triton.language as tl

@triton.jit
def s318_kernel(a_ptr, n, inc, max_out_ptr, index_out_ptr):
    # This is a sequential reduction kernel - use single thread
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    # Initialize
    k = 0
    index = 0
    max_val = tl.abs(tl.load(a_ptr))
    k += inc
    
    # Sequential loop through array
    for i in range(1, n):
        # Check bounds
        if k >= 0:
            current_val = tl.abs(tl.load(a_ptr + k))
            
            # Update max and index if current value is greater
            if current_val > max_val:
                index = i
                max_val = current_val
        
        k += inc
    
    # Store results
    tl.store(max_out_ptr, max_val)
    tl.store(index_out_ptr, index)

def s318_triton(a, inc):
    n = a.shape[0]
    
    # Output tensors
    max_out = torch.zeros(1, dtype=a.dtype, device=a.device)
    index_out = torch.zeros(1, dtype=torch.int32, device=a.device)
    
    # Launch kernel with single block
    grid = (1,)
    s318_kernel[grid](
        a, n, inc, max_out, index_out
    )
    
    max_val = max_out.item()
    index_val = index_out.item()
    chksum = max_val + float(index_val)
    
    return max_val, index_val, chksum