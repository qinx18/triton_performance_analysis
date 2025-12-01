import torch
import triton
import triton.language as tl

@triton.jit
def s318_kernel(a_ptr, inc, n_elements, output_ptr):
    BLOCK_SIZE = 256
    
    # Initialize variables
    k = 0
    index = 0
    max_val = tl.abs(tl.load(a_ptr + k))
    k += inc
    
    # Sequential search for maximum absolute value
    for i in range(1, n_elements):
        if k >= n_elements:
            break
            
        current_val = tl.abs(tl.load(a_ptr + k))
        
        # Update max and index if current value is greater
        if current_val > max_val:
            index = i
            max_val = current_val
            
        k += inc
    
    # Store results
    tl.store(output_ptr, max_val)
    tl.store(output_ptr + 1, index.to(tl.float32))

def s318_triton(a, inc):
    n_elements = a.shape[0]
    
    # Output tensor to store [max_value, index]
    output = torch.zeros(2, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single thread since this is a sequential reduction
    grid = (1,)
    
    s318_kernel[grid](
        a, inc, n_elements, output
    )
    
    max_val = output[0]
    index = int(output[1])
    chksum = max_val + float(index)
    
    return max_val, index, chksum