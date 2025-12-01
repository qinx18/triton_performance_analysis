import triton
import triton.language as tl
import torch

@triton.jit
def s318_kernel(a_ptr, inc, n_elements, 
                result_max_ptr, result_index_ptr):
    # This kernel finds the maximum absolute value and its index
    # with non-unit stride (increment)
    
    # Initialize with first element
    k = 0
    current_max = tl.abs(tl.load(a_ptr + k))
    current_index = 0
    
    # Sequential search through array with increment
    for i in range(1, n_elements):
        k += inc
        abs_val = tl.abs(tl.load(a_ptr + k))
        
        # Update if we found a larger absolute value
        if abs_val > current_max:
            current_max = abs_val
            current_index = i
    
    # Store results
    tl.store(result_max_ptr, current_max)
    tl.store(result_index_ptr, current_index)

def s318_triton(a, inc_val):
    # Get array dimensions
    n_elements = a.shape[0]
    
    # Prepare output tensors
    result_max = torch.zeros(1, dtype=a.dtype, device=a.device)
    result_index = torch.zeros(1, dtype=torch.int32, device=a.device)
    
    # Launch kernel with single thread (sequential reduction)
    grid = (1,)
    s318_kernel[grid](
        a, inc_val, n_elements,
        result_max, result_index
    )
    
    return result_max.item() + result_index.item()