import torch
import triton
import triton.language as tl

@triton.jit
def s128_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each iteration i processes sequentially due to dependencies
    # i goes from 0 to n_elements-1, where n_elements = LEN_1D/2
    
    for i in range(n_elements):
        # k = j + 1 where j starts at -1 and updates as j = k + 1
        # Iteration 0: j=-1, k=0, then j=1
        # Iteration 1: j=1, k=2, then j=3  
        # Iteration 2: j=3, k=4, then j=5
        # Pattern: k = 2*i
        k = 2 * i
        
        # Load values with bounds checking
        if k < 2 * n_elements:  # b array has same size as original arrays
            b_k = tl.load(b_ptr + k)
            c_k = tl.load(c_ptr + k)
        else:
            b_k = 0.0
            c_k = 0.0
            
        d_i = tl.load(d_ptr + i)
        
        # a[i] = b[k] - d[i]
        a_val = b_k - d_i
        tl.store(a_ptr + i, a_val)
        
        # b[k] = a[i] + c[k]  
        if k < 2 * n_elements:
            b_val = a_val + c_k
            tl.store(b_ptr + k, b_val)

def s128_triton(a, b, c, d):
    n_elements = a.shape[0] // 2
    
    # Use single thread since operations are sequential
    grid = (1,)
    BLOCK_SIZE = 1
    
    s128_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )