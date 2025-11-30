import torch
import triton
import triton.language as tl

@triton.jit
def s3110_kernel(aa_ptr, max_ptr, xindex_ptr, yindex_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # This is a reduction kernel that finds the maximum element and its indices
    # We'll use a single thread block to handle the entire 2D array
    
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Initialize with aa[0][0]
    max_val = tl.load(aa_ptr)
    max_x = 0
    max_y = 0
    
    # Sequential scan through the 2D array
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            idx = i * LEN_2D + j
            val = tl.load(aa_ptr + idx)
            
            # Update max and indices if current value is greater
            if val > max_val:
                max_val = val
                max_x = i
                max_y = j
    
    # Store results
    tl.store(max_ptr, max_val)
    tl.store(xindex_ptr, max_x)
    tl.store(yindex_ptr, max_y)

def s3110_triton(aa):
    LEN_2D = aa.shape[0]
    
    # Ensure aa is contiguous
    aa = aa.contiguous()
    
    # Output tensors
    max_val = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    xindex = torch.zeros(1, dtype=torch.int32, device=aa.device)
    yindex = torch.zeros(1, dtype=torch.int32, device=aa.device)
    
    # Launch kernel with single thread block
    grid = (1,)
    s3110_kernel[grid](
        aa, max_val, xindex, yindex,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=256
    )
    
    # Calculate chksum
    chksum = max_val.item() + float(xindex.item()) + float(yindex.item())
    
    return max_val.item() + xindex.item() + 1 + yindex.item() + 1