import triton
import triton.language as tl
import torch

@triton.jit
def s221_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Sequential dependency in b[i] = b[i-1] + ... requires serial execution
    # Process one element at a time to maintain dependency
    for i in range(1, n_elements):
        # Update a[i]
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        a_val = tl.load(a_ptr + i)
        new_a_val = a_val + c_val * d_val
        tl.store(a_ptr + i, new_a_val)
        
        # Update b[i] with dependency on b[i-1]
        b_prev = tl.load(b_ptr + i - 1)
        b_val = b_prev + new_a_val + d_val
        tl.store(b_ptr + i, b_val)

def s221_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    s221_kernel[(1,)](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )