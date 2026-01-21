import triton
import triton.language as tl
import torch

@triton.jit
def s128_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each thread block processes one iteration of the original loop
    block_id = tl.program_id(0)
    
    # Check if this block has work to do
    if block_id >= n_elements:
        return
    
    # Calculate k values for this iteration
    # j starts at -1, so k = j + 1 = 2*i, and next j = k + 1 = 2*i + 1
    i = block_id
    k = 2 * i
    
    # Load values
    d_val = tl.load(d_ptr + i)
    b_val = tl.load(b_ptr + k)
    c_val = tl.load(c_ptr + k)
    
    # Compute a[i] = b[k] - d[i]
    a_val = b_val - d_val
    
    # Store a[i]
    tl.store(a_ptr + i, a_val)
    
    # Compute b[k] = a[i] + c[k]
    b_new_val = a_val + c_val
    
    # Store b[k]
    tl.store(b_ptr + k, b_new_val)

def s128_triton(a, b, c, d):
    N = a.shape[0]
    n_elements = N // 2
    
    # Launch kernel with one thread per iteration
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, 1),)
    
    s128_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )