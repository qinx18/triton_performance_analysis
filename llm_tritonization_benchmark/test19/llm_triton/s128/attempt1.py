import triton
import triton.language as tl
import torch

@triton.jit
def s128_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Each thread block processes one iteration sequentially
    # since k values depend on previous iterations
    
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process iterations sequentially within each block
    for local_i in range(BLOCK_SIZE):
        i = block_start + local_i
        
        if i < n:
            # k = 2*i (from j=-1, k=j+1=2*i, j=k+1=2*i+1)
            k = 2 * i
            
            # a[i] = b[k] - d[i]
            b_val = tl.load(b_ptr + k)
            d_val = tl.load(d_ptr + i)
            a_val = b_val - d_val
            tl.store(a_ptr + i, a_val)
            
            # b[k] = a[i] + c[k]
            c_val = tl.load(c_ptr + k)
            b_new_val = a_val + c_val
            tl.store(b_ptr + k, b_new_val)

def s128_triton(a, b, c, d):
    N = a.shape[0]
    n = N // 2  # LEN_1D/2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s128_kernel[grid](a, b, c, d, n, BLOCK_SIZE=BLOCK_SIZE)