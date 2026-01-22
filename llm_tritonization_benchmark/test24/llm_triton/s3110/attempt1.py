import torch
import triton
import triton.language as tl

@triton.jit
def s3110_kernel(aa_ptr, max_val, max_i, max_j, LEN_2D: tl.constexpr):
    # This is a reduction operation to find max value and its indices
    # Since this is inherently sequential, we use a single thread approach
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Initialize with aa[0][0]
    current_max = tl.load(aa_ptr)
    current_max_i = 0
    current_max_j = 0
    
    # Sequential search through the 2D array
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            idx = i * LEN_2D + j
            val = tl.load(aa_ptr + idx)
            
            # Update if we found a larger value
            if val > current_max:
                current_max = val
                current_max_i = i
                current_max_j = j
    
    # Store results
    tl.store(max_val, current_max)
    tl.store(max_i, current_max_i)
    tl.store(max_j, current_max_j)

def s3110_triton(aa):
    LEN_2D = aa.shape[0]
    
    # Create output tensors
    max_val = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    max_i = torch.zeros(1, dtype=torch.int32, device=aa.device)
    max_j = torch.zeros(1, dtype=torch.int32, device=aa.device)
    
    # Launch single thread kernel
    grid = (1,)
    s3110_kernel[grid](
        aa,
        max_val,
        max_i,
        max_j,
        LEN_2D=LEN_2D
    )
    
    # Extract results and compute return value
    max_value = max_val.item()
    xindex = max_i.item()
    yindex = max_j.item()
    
    return max_value + (xindex + 1) + (yindex + 1)