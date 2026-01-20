import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, result_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    # Only use the first program
    if pid != 0:
        return
    
    # Initialize with first element
    max_val = tl.load(aa_ptr)
    max_i = 0
    max_j = 0
    
    # Sequential loop over all elements
    for i in range(N):
        for j in range(N):
            current_val = tl.load(aa_ptr + i * N + j)
            if current_val > max_val:
                max_val = current_val
                max_i = i
                max_j = j
    
    # Store results
    tl.store(result_ptr, max_val)
    tl.store(result_ptr + 1, max_i)
    tl.store(result_ptr + 2, max_j)

def s13110_triton(aa):
    N = aa.shape[0]
    
    # Allocate result tensor for max_val, max_i, max_j
    result = torch.zeros(3, dtype=aa.dtype, device=aa.device)
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s13110_kernel[grid](
        aa,
        result,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    max_val = result[0]
    xindex = int(result[1])
    yindex = int(result[2])
    
    return max_val + (xindex + 1) + (yindex + 1)