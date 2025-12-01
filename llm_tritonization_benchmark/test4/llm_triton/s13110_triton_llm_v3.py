import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, LEN_2D: tl.constexpr, max_out_ptr, xindex_out_ptr, yindex_out_ptr):
    # This kernel finds the maximum element and its indices in a 2D array
    # Each program handles the entire 2D array to find global maximum
    
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    # Initialize with first element
    max_val = tl.load(aa_ptr)  # aa[0][0]
    xindex = 0
    yindex = 0
    
    # Sequential search through the entire 2D array
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            idx = i * LEN_2D + j
            val = tl.load(aa_ptr + idx)
            
            # Update if we found a larger value
            if val > max_val:
                max_val = val
                xindex = i
                yindex = j
    
    # Store results
    tl.store(max_out_ptr, max_val)
    tl.store(xindex_out_ptr, xindex)
    tl.store(yindex_out_ptr, yindex)

def s13110_triton(aa):
    LEN_2D = aa.shape[0]
    
    # Output tensors
    max_out = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    xindex_out = torch.zeros(1, dtype=torch.int32, device=aa.device)
    yindex_out = torch.zeros(1, dtype=torch.int32, device=aa.device)
    
    # Launch kernel with single program
    grid = (1,)
    s13110_kernel[grid](
        aa, LEN_2D,
        max_out, xindex_out, yindex_out
    )
    
    max_val = max_out.item()
    xindex = xindex_out.item()
    yindex = yindex_out.item()
    
    chksum = max_val + float(xindex) + float(yindex)
    
    return max_val + xindex + 1 + yindex + 1