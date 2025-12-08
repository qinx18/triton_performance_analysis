import torch
import triton
import triton.language as tl

@triton.jit
def s128_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Sequential computation due to data dependencies
    # Each thread block processes a contiguous range of iterations
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        
        # Use conditional to handle bounds checking without break
        if idx < n_elements:
            # k = j + 1 where j starts at -1 and increments by 2 each iteration
            # j = -1, 1, 3, 5, ... => j = 2*i - 1
            # k = j + 1 = 2*i
            k = 2 * idx
            
            # Load values
            if k < n_elements:
                b_k = tl.load(b_ptr + k)
                c_k = tl.load(c_ptr + k)
            else:
                b_k = 0.0
                c_k = 0.0
                
            d_i = tl.load(d_ptr + idx)
            
            # a[i] = b[k] - d[i]
            a_val = b_k - d_i
            tl.store(a_ptr + idx, a_val)
            
            # b[k] = a[i] + c[k]
            if k < n_elements:
                b_val = a_val + c_k
                tl.store(b_ptr + k, b_val)

def s128_triton(a, b, c, d):
    n_elements = a.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s128_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )