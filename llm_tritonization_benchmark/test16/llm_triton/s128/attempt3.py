import torch
import triton
import triton.language as tl

@triton.jit
def s128_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Each program handles one element due to sequential dependencies
    idx = pid
    
    if idx >= n_elements:
        return
        
    # Trace through the loop:
    # i=0: j=-1, k=j+1=0, j=k+1=1
    # i=1: j=1,  k=j+1=2, j=k+1=3  
    # i=2: j=3,  k=j+1=4, j=k+1=5
    # So k = 2*idx
    k = 2 * idx
    
    # Load individual values
    b_k = tl.load(b_ptr + k)
    c_k = tl.load(c_ptr + k)
    d_i = tl.load(d_ptr + idx)
    
    # a[i] = b[k] - d[i]
    a_val = b_k - d_i
    tl.store(a_ptr + idx, a_val)
    
    # b[k] = a[i] + c[k]
    b_val = a_val + c_k
    tl.store(b_ptr + k, b_val)

def s128_triton(a, b, c, d):
    n_elements = a.shape[0] // 2
    
    grid = (n_elements,)
    
    s128_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=1
    )