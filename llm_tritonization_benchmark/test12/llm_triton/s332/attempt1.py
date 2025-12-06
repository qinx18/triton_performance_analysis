import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, t, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    index = -2
    value = -1.0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Check if any element in this block is greater than t
        condition = a_vals > t
        
        # Process each element in the block
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                if condition[i]:
                    index = block_start + i
                    value = a_vals[i]
                    # Found first value, store results and exit
                    chksum = value + index
                    tl.store(result_ptr, value)
                    tl.store(result_ptr + 1, index)
                    tl.store(result_ptr + 2, chksum)
                    return
    
    # If no value found, store defaults
    chksum = value + index
    tl.store(result_ptr, value)
    tl.store(result_ptr + 1, index)
    tl.store(result_ptr + 2, chksum)

def s332_triton(a, t):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create result tensor to store value, index, and chksum
    result = torch.zeros(3, dtype=torch.float32, device=a.device)
    
    # Launch kernel with single block since we need sequential search
    s332_kernel[(1,)](
        a, t, result, n_elements, BLOCK_SIZE
    )
    
    return result[0].item()  # Return the value