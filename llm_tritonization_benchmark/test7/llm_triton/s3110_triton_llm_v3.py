import torch
import triton
import triton.language as tl

@triton.jit
def s3110_kernel(aa_ptr, max_ptr, xindex_ptr, yindex_ptr, LEN_2D: tl.constexpr):
    # This kernel finds the maximum element and its indices in a 2D array
    # Each block handles the entire 2D array sequentially
    
    block_id = tl.program_id(0)
    if block_id > 0:
        return
    
    # Initialize with first element
    max_val = tl.load(aa_ptr)
    max_x = 0
    max_y = 0
    
    # Sequential scan through the 2D array
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            offset = i * LEN_2D + j
            val = tl.load(aa_ptr + offset)
            
            # Update max and indices if we find a larger value
            is_greater = val > max_val
            max_val = tl.where(is_greater, val, max_val)
            max_x = tl.where(is_greater, i, max_x)
            max_y = tl.where(is_greater, j, max_y)
    
    # Store results
    tl.store(max_ptr, max_val)
    tl.store(xindex_ptr, max_x)
    tl.store(yindex_ptr, max_y)

def s3110_triton(aa):
    # Create output tensors for results
    max_val = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    xindex = torch.zeros(1, dtype=torch.int32, device=aa.device)
    yindex = torch.zeros(1, dtype=torch.int32, device=aa.device)
    
    LEN_2D = aa.shape[0]
    
    # Launch kernel with single block
    grid = (1,)
    s3110_kernel[grid](
        aa, max_val, xindex, yindex,
        LEN_2D=LEN_2D
    )
    
    return max_val.item() + (xindex.item() + 1) + (yindex.item() + 1)