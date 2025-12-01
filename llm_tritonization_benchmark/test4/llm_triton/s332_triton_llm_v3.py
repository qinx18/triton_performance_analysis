import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, t, result_ptr, n):
    BLOCK_SIZE = 256
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    index = -2
    value = -1.0
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Check if any value in this block is greater than t
        condition = a_vals > t
        
        if tl.any(condition):
            # Find the first index where condition is true
            for i in range(BLOCK_SIZE):
                if block_start + i >= n:
                    break
                offset = block_start + i
                a_val = tl.load(a_ptr + offset)
                if a_val > t:
                    index = offset
                    value = a_val
                    break
            break
    
    chksum = value + index
    tl.store(result_ptr, chksum)

def s332_triton(a, b, c, d, e, aa, bb, cc, t):
    n = a.shape[0]
    result = torch.zeros(1, dtype=torch.float32, device=a.device)
    
    grid = (1,)
    s332_kernel[grid](
        a, t, result, n
    )
    
    return result.item()