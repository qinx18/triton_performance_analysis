import triton
import triton.language as tl
import torch

@triton.jit
def s318_kernel(a_ptr, n_elements, inc, result_ptr):
    # Find maximum absolute value and its index with stride
    current_max = tl.abs(tl.load(a_ptr))  # max = ABS(a[0])
    max_index = 0
    
    # Sequential search with stride
    k = inc
    for i in range(1, n_elements):
        if k < n_elements:
            val = tl.abs(tl.load(a_ptr + k))
            # if (ABS(a[k]) > max)
            if val > current_max:
                max_index = i
                current_max = val
        k += inc
    
    # Store results: max + index + 1 (as per return statement)
    result = current_max + max_index + 1
    tl.store(result_ptr, result)

def s318_triton(a, inc):
    n_elements = a.shape[0]
    
    # Output tensor for result
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single thread (sequential reduction)
    grid = (1,)
    s318_kernel[grid](
        a, n_elements, inc, result
    )
    
    return result.item()