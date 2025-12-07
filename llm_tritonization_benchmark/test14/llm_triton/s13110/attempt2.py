import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, result_ptr, LEN_2D: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:
        # Initialize with aa[0][0]
        max_val = tl.load(aa_ptr)
        xindex = 0
        yindex = 0
        
        # Sequential loop over all elements
        for i in range(LEN_2D):
            for j in range(LEN_2D):
                idx = i * LEN_2D + j
                val = tl.load(aa_ptr + idx)
                
                if val > max_val:
                    max_val = val
                    xindex = i
                    yindex = j
        
        # Store result as max + xindex + yindex
        chksum = max_val + xindex + yindex
        tl.store(result_ptr, chksum)

def s13110_triton(aa):
    LEN_2D = aa.shape[0]
    
    # Output tensor
    result = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    
    # Launch kernel
    grid = (1,)
    s13110_kernel[grid](
        aa,
        result,
        LEN_2D=LEN_2D
    )
    
    return result.item()