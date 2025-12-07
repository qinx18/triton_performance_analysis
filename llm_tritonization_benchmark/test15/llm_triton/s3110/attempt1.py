import torch
import triton
import triton.language as tl

@triton.jit
def s3110_kernel(
    aa_ptr,
    result_ptr,
    LEN_2D: tl.constexpr,
):
    # This is an argmax reduction - find max value and its indices
    # We'll do this sequentially since it's inherently a reduction operation
    
    # Initialize with aa[0][0]
    max_val = tl.load(aa_ptr)
    max_i = 0
    max_j = 0
    
    # Sequential search through the 2D array
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            idx = i * LEN_2D + j
            val = tl.load(aa_ptr + idx)
            if val > max_val:
                max_val = val
                max_i = i
                max_j = j
    
    # Store results
    tl.store(result_ptr, max_val)
    tl.store(result_ptr + 1, max_i.to(tl.float32))
    tl.store(result_ptr + 2, max_j.to(tl.float32))

def s3110_triton(aa):
    LEN_2D = aa.shape[0]
    
    # Create result tensor to store max_val, max_i, max_j
    result = torch.zeros(3, dtype=torch.float32, device=aa.device)
    
    # Launch single thread since this is a global reduction
    grid = (1,)
    
    s3110_kernel[grid](
        aa,
        result,
        LEN_2D,
    )
    
    max_val = result[0]
    xindex = int(result[1])
    yindex = int(result[2])
    
    # Calculate chksum (though not used in return)
    chksum = max_val + float(xindex) + float(yindex)
    
    # Return as specified in C code: max + xindex+1 + yindex+1
    return max_val + (xindex + 1) + (yindex + 1)