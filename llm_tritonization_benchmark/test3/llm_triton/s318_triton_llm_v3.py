import triton
import triton.language as tl
import torch

@triton.jit
def s318_kernel(a_ptr, output_ptr, inc, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel finds the maximum absolute value and its index
    # Each block processes the entire array independently
    block_id = tl.program_id(0)
    
    if block_id > 0:
        return
    
    # Initialize with first element
    k = 0
    max_val = tl.abs(tl.load(a_ptr + k))
    max_index = 0
    k += inc
    
    # Sequential search through array
    for i in range(1, n_elements):
        if k < n_elements:
            curr_val = tl.abs(tl.load(a_ptr + k))
            if curr_val > max_val:
                max_val = curr_val
                max_index = i
        k += inc
    
    # Store results
    tl.store(output_ptr, max_val)
    tl.store(output_ptr + 1, max_index.to(tl.float32))

def s318_triton(a, inc=1):
    n_elements = a.shape[0]
    
    # Output tensor to store max value and index
    output = torch.zeros(2, dtype=torch.float32, device=a.device)
    
    BLOCK_SIZE = 1
    grid = (1,)
    
    s318_kernel[grid](
        a, output, inc, n_elements, BLOCK_SIZE
    )
    
    max_val = output[0].item()
    max_index = int(output[1].item())
    
    return max_val + max_index + 1