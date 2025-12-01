import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, max_ptr, xindex_ptr, yindex_ptr, LEN_2D: tl.constexpr):
    # This kernel finds the maximum value and its indices in a 2D array
    # Since this is a global reduction, we use a single block
    
    # Initialize with first element
    max_val = tl.load(aa_ptr)
    max_x = 0
    max_y = 0
    
    # Sequential scan through the 2D array
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            idx = i * LEN_2D + j
            val = tl.load(aa_ptr + idx)
            
            # Update maximum if current value is greater
            if val > max_val:
                max_val = val
                max_x = i
                max_y = j
    
    # Store results
    tl.store(max_ptr, max_val)
    tl.store(xindex_ptr, max_x)
    tl.store(yindex_ptr, max_y)

def s13110_triton(aa):
    LEN_2D = int(aa.shape[0])
    
    # Output tensors
    max_result = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    xindex_result = torch.zeros(1, dtype=torch.int32, device=aa.device)
    yindex_result = torch.zeros(1, dtype=torch.int32, device=aa.device)
    
    # Launch kernel with single block since this is a global reduction
    grid = (1,)
    
    s13110_kernel[grid](
        aa, max_result, xindex_result, yindex_result,
        LEN_2D=LEN_2D
    )
    
    max_val = max_result.item()
    xindex = xindex_result.item()
    yindex = yindex_result.item()
    
    chksum = max_val + float(xindex) + float(yindex)
    
    return max_val + xindex + 1 + yindex + 1