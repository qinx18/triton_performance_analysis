import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, max_ptr, xindex_ptr, yindex_ptr, LEN_2D: tl.constexpr):
    # Initialize values
    max_val = tl.load(aa_ptr)  # aa[0][0]
    xindex = 0
    yindex = 0
    
    # Sequential scan through all elements
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
    
    # Output tensors for results
    max_result = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    xindex_result = torch.zeros(1, dtype=torch.int32, device=aa.device)
    yindex_result = torch.zeros(1, dtype=torch.int32, device=aa.device)
    
    # Launch kernel with single thread
    grid = (1,)
    s13110_kernel[grid](
        aa, max_result, xindex_result, yindex_result, LEN_2D
    )
    
    # Return results (adding 1 to indices as in original C code)
    max_val = max_result.item()
    xindex = xindex_result.item()
    yindex = yindex_result.item()
    
    return max_val + (xindex + 1) + (yindex + 1)