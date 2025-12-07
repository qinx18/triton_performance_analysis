import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, result_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:  # Only one thread block does the work
        # Initialize with first element
        max_val = tl.load(aa_ptr)  # aa[0][0]
        max_i = 0
        max_j = 0
        
        # Process entire 2D array sequentially
        for idx in range(LEN_2D * LEN_2D):
            i = idx // LEN_2D
            j = idx % LEN_2D
            val = tl.load(aa_ptr + idx)
            
            # Update max if current value is greater
            is_greater = val > max_val
            max_val = tl.where(is_greater, val, max_val)
            max_i = tl.where(is_greater, i, max_i)
            max_j = tl.where(is_greater, j, max_j)
        
        # Store results
        tl.store(result_ptr, max_val)
        tl.store(result_ptr + 1, max_i)
        tl.store(result_ptr + 2, max_j)

def s13110_triton(aa):
    LEN_2D = aa.shape[0]
    
    # Create result tensor to store max, xindex, yindex
    result = torch.zeros(3, dtype=aa.dtype, device=aa.device)
    
    # Launch single thread block since this is a global reduction
    grid = (1,)
    BLOCK_SIZE = 64
    
    s13110_kernel[grid](
        aa, result, 
        LEN_2D=LEN_2D, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    max_val = result[0].item()
    xindex = int(result[1].item())
    yindex = int(result[2].item())
    
    return max_val + xindex + 1 + yindex + 1