import torch
import triton
import triton.language as tl

@triton.jit
def s128_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx >= n_elements:
            break
            
        # k = j + 1, where j starts at -1 and increments by 2 each iteration
        # So j = -1, 1, 3, 5, ... and k = 0, 2, 4, 6, ...
        k = 2 * idx
        
        # Load values
        b_k = tl.load(b_ptr + k)
        d_i = tl.load(d_ptr + idx)
        c_k = tl.load(c_ptr + k)
        
        # a[i] = b[k] - d[i]
        a_val = b_k - d_i
        tl.store(a_ptr + idx, a_val)
        
        # b[k] = a[i] + c[k]
        b_val = a_val + c_k
        tl.store(b_ptr + k, b_val)

def s128_triton(a, b, c, d):
    n_elements = a.shape[0] // 2
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s128_kernel[grid](a, b, c, d, n_elements, BLOCK_SIZE)