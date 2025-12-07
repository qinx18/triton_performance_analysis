import triton
import triton.language as tl
import torch

@triton.jit
def s318_kernel(a_ptr, n_elements, inc, max_out_ptr, index_out_ptr):
    # This kernel finds the maximum absolute value and its index with stride
    # Each block processes the entire array to find global max
    pid = tl.program_id(0)
    
    # Only process with first block
    if pid != 0:
        return
    
    # Initialize with first element
    k = 0
    current_max = tl.abs(tl.load(a_ptr + k))
    current_index = 0
    k += inc
    
    # Process remaining elements
    for i in range(1, n_elements):
        val = tl.abs(tl.load(a_ptr + k))
        # Update max and index only if we found a strictly larger value
        if val > current_max:
            current_max = val
            current_index = i
        k += inc
    
    # Store results
    tl.store(max_out_ptr, current_max)
    tl.store(index_out_ptr, current_index)

def s318_triton(a, inc):
    n_elements = a.shape[0]
    
    # Output tensors for max value and index
    max_out = torch.zeros(1, dtype=a.dtype, device=a.device)
    index_out = torch.zeros(1, dtype=torch.int32, device=a.device)
    
    # Launch kernel with single block
    grid = (1,)
    s318_kernel[grid](
        a, n_elements, inc,
        max_out, index_out
    )
    
    # Return the results
    max_val = max_out.item()
    index_val = index_out.item()
    
    return max_val + index_val + 1