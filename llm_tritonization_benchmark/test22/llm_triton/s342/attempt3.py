import triton
import triton.language as tl
import torch

@triton.jit
def s342_kernel(a_ptr, b_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid > 0:
        return
    
    # Process all elements sequentially in a single thread block
    j = -1
    for i in range(n):
        a_val = tl.load(a_ptr + i)
        if a_val > 0.0:
            j = j + 1
            b_val = tl.load(b_ptr + j)
            tl.store(result_ptr + i, b_val)
        else:
            tl.store(result_ptr + i, a_val)

def s342_triton(a, b):
    n = a.shape[0]
    result = torch.empty_like(a)
    
    grid = (1,)
    s342_kernel[grid](a, b, result, n, BLOCK_SIZE=1)
    
    a.copy_(result)
    return a