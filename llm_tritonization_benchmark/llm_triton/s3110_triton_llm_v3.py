import triton
import triton.language as tl
import torch

@triton.jit
def s3110_kernel(aa_ptr, max_ptr, xindex_ptr, yindex_ptr, LEN_2D: tl.constexpr):
    # This is a reduction kernel that finds the maximum element and its indices
    # We'll use a single block to handle the entire 2D array
    
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Initialize with first element
    current_max = tl.load(aa_ptr)
    current_xindex = 0
    current_yindex = 0
    
    # Iterate through all elements
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            offset = i * LEN_2D + j
            value = tl.load(aa_ptr + offset)
            
            # Update max and indices if we found a larger value
            if value > current_max:
                current_max = value
                current_xindex = i
                current_yindex = j
    
    # Store results
    tl.store(max_ptr, current_max)
    tl.store(xindex_ptr, current_xindex)
    tl.store(yindex_ptr, current_yindex)

def s3110_triton(aa):
    LEN_2D = aa.shape[0]
    
    # Create output tensors
    max_result = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    xindex_result = torch.zeros(1, dtype=torch.int32, device=aa.device)
    yindex_result = torch.zeros(1, dtype=torch.int32, device=aa.device)
    
    # Launch kernel with single block
    grid = (1,)
    s3110_kernel[grid](
        aa, max_result, xindex_result, yindex_result, LEN_2D
    )
    
    max_val = max_result.item()
    xindex = xindex_result.item()
    yindex = yindex_result.item()
    chksum = max_val + float(xindex) + float(yindex)
    
    return max_val + xindex + 1 + yindex + 1