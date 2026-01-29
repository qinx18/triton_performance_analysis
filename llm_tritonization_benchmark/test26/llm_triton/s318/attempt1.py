import triton
import triton.language as tl
import torch

@triton.jit
def s318_kernel(a_ptr, inc, n_elements, output_ptr):
    # This is an argmax reduction that needs to be done sequentially
    # due to the stride pattern and comparison logic
    
    k = 0
    index = 0
    
    # Load first element
    first_val = tl.load(a_ptr + k)
    max_val = tl.abs(first_val)
    k += inc
    
    # Sequential loop to find max absolute value and its index
    for i in range(1, n_elements):
        if k < n_elements:
            current_val = tl.load(a_ptr + k)
            abs_val = tl.abs(current_val)
            
            # Update if current absolute value is greater than max
            if abs_val > max_val:
                index = i
                max_val = abs_val
        
        k += inc
    
    # Store results
    tl.store(output_ptr, max_val)
    tl.store(output_ptr + 1, index.to(tl.float32))

def s318_triton(a, inc=1):
    n_elements = a.shape[0]
    
    # Output tensor to store [max_val, index]
    output = torch.zeros(2, dtype=torch.float32, device=a.device)
    
    # Launch kernel with single thread since this is inherently sequential
    grid = (1,)
    s318_kernel[grid](
        a, inc, n_elements, output
    )
    
    max_val = output[0]
    index = int(output[1])
    
    return max_val + index + 1