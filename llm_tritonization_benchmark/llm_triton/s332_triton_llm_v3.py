import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, output_ptr, t, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each block handles one search operation
    block_id = tl.program_id(0)
    if block_id > 0:
        return
    
    # Initialize search variables
    found_index = -2
    found_value = -1.0
    
    # Sequential search through the array
    for block_start in range(0, n_elements, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load values from array a
        values = tl.load(a_ptr + offsets, mask=mask, other=float('-inf'))
        
        # Check if any values in this block are greater than threshold
        greater_mask = values > t
        
        if tl.any(greater_mask):
            # Find the first index that satisfies the condition
            for i in range(BLOCK_SIZE):
                if block_start + i < n_elements:
                    offset = block_start + i
                    value = tl.load(a_ptr + offset)
                    if value > t:
                        found_index = offset
                        found_value = value
                        # Store results and exit
                        tl.store(output_ptr, found_value)
                        tl.store(output_ptr + 1, found_index.to(tl.float32))
                        return
    
    # If no value found, store the default values
    tl.store(output_ptr, found_value)
    tl.store(output_ptr + 1, found_index.to(tl.float32))

def s332_triton(a, t_value):
    n_elements = a.numel()
    
    # Output tensor to store [value, index]
    output = torch.zeros(2, device=a.device, dtype=a.dtype)
    
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s332_kernel[grid](
        a, output, t_value, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    value = output[0].item()
    index = int(output[1].item())
    chksum = value + float(index)
    
    return value, index, chksum