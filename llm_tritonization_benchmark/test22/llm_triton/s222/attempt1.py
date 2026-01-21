import triton
import triton.language as tl
import torch

@triton.jit
def s222_kernel(a_ptr, b_ptr, c_ptr, e_ptr, N, BLOCK_SIZE: tl.constexpr):
    # This kernel must be run with a single thread due to strict sequential dependency
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    # Process sequentially from i=1 to N-1
    for i in range(1, N):
        # Load b[i], c[i], a[i], e[i-1]
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        a_val = tl.load(a_ptr + i)
        e_prev = tl.load(e_ptr + (i - 1))
        
        # a[i] += b[i] * c[i]
        bc_product = b_val * c_val
        a_val = a_val + bc_product
        tl.store(a_ptr + i, a_val)
        
        # e[i] = e[i - 1] * e[i - 1]
        e_new = e_prev * e_prev
        tl.store(e_ptr + i, e_new)
        
        # a[i] -= b[i] * c[i]
        a_val = a_val - bc_product
        tl.store(a_ptr + i, a_val)

def s222_triton(a, b, c, e):
    N = a.shape[0]
    
    BLOCK_SIZE = 1024
    grid = (1,)  # Single thread only due to sequential dependency
    
    s222_kernel[grid](
        a, b, c, e,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )