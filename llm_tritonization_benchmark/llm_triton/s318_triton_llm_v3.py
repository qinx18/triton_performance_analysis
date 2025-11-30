import triton
import triton.language as tl
import torch

@triton.jit
def s318_kernel(a_ptr, output_ptr, inc, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel finds the maximum absolute value and its index with non-unit stride
    # Each program handles the entire array (reduction operation)
    program_id = tl.program_id(0)
    
    if program_id > 0:
        return
    
    # Initialize with first element
    k = 0
    index = 0
    max_val = tl.abs(tl.load(a_ptr + k))
    k += inc
    
    # Sequential search for maximum absolute value
    for i in range(1, n_elements):
        if k >= n_elements * inc:
            break
            
        current_abs = tl.abs(tl.load(a_ptr + k))
        
        # Update max and index if current value is greater
        if current_abs > max_val:
            max_val = current_abs
            index = i
            
        k += inc
    
    # Store results
    tl.store(output_ptr, max_val)
    tl.store(output_ptr + 1, index.to(tl.float32))

def s318_triton(a, inc=1):
    n_elements = a.shape[0]
    
    # Output tensor to store max value and index
    output = torch.zeros(2, dtype=torch.float32, device=a.device)
    
    BLOCK_SIZE = 1024
    
    # Launch kernel with single thread block since this is a sequential reduction
    grid = (1,)
    s318_kernel[grid](
        a, output, inc, n_elements, BLOCK_SIZE
    )
    
    max_val = output[0].item()
    index = int(output[1].item())
    chksum = max_val + float(index)
    
    return max_val + index + 1