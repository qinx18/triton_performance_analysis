import torch
import triton
import triton.language as tl

@triton.jit
def s318_kernel(a_ptr, max_val_ptr, max_idx_ptr, n_elements, inc, BLOCK_SIZE: tl.constexpr):
    # This kernel finds the maximum absolute value and its index with stride inc
    # Each program handles the entire reduction since it's a global operation
    
    pid = tl.program_id(0)
    if pid > 0:  # Only process with first program
        return
    
    # Initialize with first element
    k = 0
    current_max = tl.abs(tl.load(a_ptr + k))
    max_index = 0
    k += inc
    
    # Process remaining elements sequentially
    for i in range(1, n_elements):
        # Safety check - exit if we would go out of bounds
        valid = k < n_elements * inc
        if not valid:
            return
            
        # Load current value
        current_val = tl.abs(tl.load(a_ptr + k))
        
        # Update max and index if current value is greater
        if current_val > current_max:
            current_max = current_val
            max_index = i
            
        k += inc
    
    # Store results
    tl.store(max_val_ptr, current_max)
    tl.store(max_idx_ptr, max_index)

def s318_triton(a, inc):
    n_elements = a.shape[0]
    
    # Output tensors for max value and index
    max_val = torch.zeros(1, dtype=a.dtype, device=a.device)
    max_idx = torch.zeros(1, dtype=torch.int32, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)  # Single program handles the entire reduction
    
    s318_kernel[grid](
        a, max_val, max_idx,
        n_elements, inc,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return chksum equivalent (max + index)
    return max_val.item() + float(max_idx.item())