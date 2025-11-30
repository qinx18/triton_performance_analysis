import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, t, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a search operation that finds the first element > threshold
    # Each program handles one search operation
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    # Initialize result values
    index = -2
    value = -1.0
    found = False
    
    # Sequential search through the array
    for block_start in range(0, n_elements, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load values
        vals = tl.load(a_ptr + offsets, mask=mask, other=float('-inf'))
        
        # Check which elements are greater than threshold
        greater_mask = vals > t
        
        if tl.any(greater_mask & mask):
            # Find the first element that satisfies condition in this block
            for i in range(BLOCK_SIZE):
                if block_start + i < n_elements:
                    val = tl.load(a_ptr + block_start + i)
                    if val > t:
                        index = block_start + i
                        value = val
                        found = True
                        break
            if found:
                break
    
    # Store results
    chksum = value + tl.cast(index, tl.float32)
    tl.store(result_ptr, chksum)
    tl.store(result_ptr + 1, value)
    tl.store(result_ptr + 2, tl.cast(index, tl.float32))

def s332_triton(a, t_val):
    n_elements = a.shape[0]
    
    # Create result tensor to store [chksum, value, index]
    result = torch.zeros(3, dtype=torch.float32, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)  # Single program since this is a sequential search
    
    s332_kernel[grid](
        a, t_val, result, n_elements, BLOCK_SIZE
    )
    
    return result[1].item()  # Return the value found