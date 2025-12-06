import torch
import triton
import triton.language as tl

@triton.jit
def s221_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    i,
    BLOCK_SIZE: tl.constexpr,
):
    # Process single element at index i
    pid = tl.program_id(0)
    
    if pid == 0:  # Only one thread processes this element
        # Load values
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        a_val = tl.load(a_ptr + i)
        b_prev_val = tl.load(b_ptr + i - 1)
        
        # Compute: a[i] += c[i] * d[i]
        new_a_val = a_val + c_val * d_val
        
        # Compute: b[i] = b[i-1] + a[i] + d[i]
        new_b_val = b_prev_val + new_a_val + d_val
        
        # Store results
        tl.store(a_ptr + i, new_a_val)
        tl.store(b_ptr + i, new_b_val)

def s221_triton(a, b, c, d):
    LEN_1D = a.shape[0]
    
    # Process each element sequentially due to loop-carried dependency
    for i in range(1, LEN_1D):
        s221_kernel[(1,)](
            a, b, c, d,
            i,
            BLOCK_SIZE=1,
        )