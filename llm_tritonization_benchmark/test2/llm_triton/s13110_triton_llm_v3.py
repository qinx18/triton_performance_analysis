import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, max_ptr, xindex_ptr, yindex_ptr, LEN_2D: tl.constexpr):
    # This is a reduction kernel that finds max element and its indices
    # Use single thread since this is inherently sequential
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Initialize
    max_val = tl.load(aa_ptr)  # aa[0][0]
    xindex = 0
    yindex = 0
    
    # Sequential search through 2D array
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            idx = i * LEN_2D + j
            val = tl.load(aa_ptr + idx)
            if val > max_val:
                max_val = val
                xindex = i
                yindex = j
    
    # Store results
    tl.store(max_ptr, max_val)
    tl.store(xindex_ptr, xindex)
    tl.store(yindex_ptr, yindex)

def s13110_triton(aa):
    LEN_2D = aa.shape[0]
    
    # Create output tensors
    max_tensor = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    xindex_tensor = torch.zeros(1, dtype=torch.int32, device=aa.device)
    yindex_tensor = torch.zeros(1, dtype=torch.int32, device=aa.device)
    
    # Launch kernel with single thread
    grid = (1,)
    s13110_kernel[grid](
        aa, max_tensor, xindex_tensor, yindex_tensor,
        LEN_2D=LEN_2D
    )
    
    max_val = max_tensor.item()
    xindex = xindex_tensor.item()
    yindex = yindex_tensor.item()
    
    chksum = max_val + float(xindex) + float(yindex)
    
    return max_val + xindex + 1 + yindex + 1