import triton
import triton.language as tl
import torch

@triton.jit
def s13110_kernel(aa_ptr, result_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Only use first thread of first block to do the sequential computation
    if pid == 0:
        # Initialize with aa[0][0]
        max_val = tl.load(aa_ptr)
        max_i = 0
        max_j = 0
        
        # Sequential nested loops
        for i in range(len_2d):
            for j in range(len_2d):
                idx = i * len_2d + j
                val = tl.load(aa_ptr + idx)
                if val > max_val:
                    max_val = val
                    max_i = i
                    max_j = j
        
        # Store result: max + (xindex+1) + (yindex+1)
        result = max_val + tl.cast(max_i + 1, tl.float32) + tl.cast(max_j + 1, tl.float32)
        tl.store(result_ptr, result)

def s13110_triton(aa, len_2d):
    result = torch.zeros(1, dtype=torch.float32, device=aa.device)
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s13110_kernel[grid](
        aa,
        result,
        len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()