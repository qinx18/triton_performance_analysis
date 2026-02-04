import triton
import triton.language as tl
import torch

@triton.jit
def s128_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each block processes a chunk of the loop iterations
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements in this block
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx >= n_elements:
            break
            
        # k = j + 1, where j starts at -1 and increments by 2 each iteration
        # So for iteration i: j = -1 + 2*i, k = 2*i
        k = 2 * idx
        
        # Check bounds for k
        if k >= n_elements * 2:
            break
            
        # a[i] = b[k] - d[i]
        b_val = tl.load(b_ptr + k)
        d_val = tl.load(d_ptr + idx)
        a_val = b_val - d_val
        tl.store(a_ptr + idx, a_val)
        
        # b[k] = a[i] + c[k]
        c_val = tl.load(c_ptr + k)
        b_new = a_val + c_val
        tl.store(b_ptr + k, b_new)

def s128_triton(a, b, c, d):
    N = a.shape[0]
    n_elements = N // 2  # Loop runs for LEN_1D/2 iterations
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s128_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )