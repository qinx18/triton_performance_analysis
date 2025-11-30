import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(
    aa_ptr,
    max_ptr,
    xindex_ptr,
    yindex_ptr,
    LEN_2D: tl.constexpr,
):
    # Single program handles the entire reduction
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Initialize with aa[0][0]
    max_val = tl.load(aa_ptr)
    xindex_val = 0
    yindex_val = 0
    
    # Sequential scan through the 2D array
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            offset = i * LEN_2D + j
            val = tl.load(aa_ptr + offset)
            
            # Update max and indices if current value is greater
            if val > max_val:
                max_val = val
                xindex_val = i
                yindex_val = j
    
    # Store results
    tl.store(max_ptr, max_val)
    tl.store(xindex_ptr, xindex_val)
    tl.store(yindex_ptr, yindex_val)

def s13110_triton(aa):
    LEN_2D = aa.shape[0]
    
    # Output tensors
    max_out = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    xindex_out = torch.zeros(1, dtype=torch.int32, device=aa.device)
    yindex_out = torch.zeros(1, dtype=torch.int32, device=aa.device)
    
    # Launch kernel with single thread
    grid = (1,)
    s13110_kernel[grid](
        aa,
        max_out,
        xindex_out,
        yindex_out,
        LEN_2D,
    )
    
    # Calculate chksum
    chksum = max_out.item() + float(xindex_out.item()) + float(yindex_out.item())
    
    return max_out.item() + xindex_out.item() + 1 + yindex_out.item() + 1