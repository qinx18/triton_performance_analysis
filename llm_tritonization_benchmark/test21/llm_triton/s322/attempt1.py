import triton
import triton.language as tl
import torch

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel must process the entire array sequentially
    # due to strict loop-carried dependencies
    pid = tl.program_id(0)
    
    # Only one program should run to maintain sequential dependency
    if pid != 0:
        return
    
    # Process sequentially from i=2 to n-1
    for i in range(2, n):
        # Load a[i-2], a[i-1], a[i], b[i], c[i]
        a_i_minus_2 = tl.load(a_ptr + (i - 2))
        a_i_minus_1 = tl.load(a_ptr + (i - 1))
        a_i = tl.load(a_ptr + i)
        b_i = tl.load(b_ptr + i)
        c_i = tl.load(c_ptr + i)
        
        # Compute: a[i] = a[i] + a[i-1] * b[i] + a[i-2] * c[i]
        result = a_i + a_i_minus_1 * b_i + a_i_minus_2 * c_i
        
        # Store result
        tl.store(a_ptr + i, result)

def s322_triton(a, b, c):
    n = a.shape[0]
    
    # Use single program since computation must be sequential
    grid = (1,)
    BLOCK_SIZE = 256
    
    s322_kernel[grid](a, b, c, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return a