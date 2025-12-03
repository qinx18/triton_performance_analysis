import triton
import triton.language as tl
import torch

@triton.jit
def s318_kernel(a_ptr, result_ptr, inc, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel finds the maximum absolute value and its index
    # Each block processes the entire array sequentially
    block_id = tl.program_id(0)
    
    if block_id > 0:
        return
    
    # Initialize with first element
    k = 0
    index = 0
    first_val = tl.load(a_ptr + k)
    max_val = tl.abs(first_val)
    k += inc
    
    # Process remaining elements sequentially
    for i in range(1, n_elements):
        current_val = tl.load(a_ptr + k)
        abs_val = tl.abs(current_val)
        
        # Update max and index if current is greater
        is_greater = abs_val > max_val
        max_val = tl.where(is_greater, abs_val, max_val)
        index = tl.where(is_greater, i, index)
        
        k += inc
    
    # Store results
    tl.store(result_ptr, max_val)
    tl.store(result_ptr + 1, index)

def s318_triton(a, inc):
    n_elements = a.shape[0]
    
    # Create output tensor for max value and index
    result = torch.zeros(2, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s318_kernel[grid](
        a, result, inc, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    max_val = result[0].item()
    index = int(result[1].item())
    chksum = max_val + float(index)
    
    return chksum