import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_half,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each thread handles one i value
    i = pid
    if i >= n_half:
        return
        
    # Load values for this i
    b_val = tl.load(b_ptr + i)
    c_val = tl.load(c_ptr + i)
    d_val = tl.load(d_ptr + i)
    e_val = tl.load(e_ptr + i)
    
    # Calculate j values - j starts at -1, then increments
    # For i=0: j becomes 0, then possibly 1
    # For i=k: j becomes 2*k, then possibly 2*k+1
    j_base = i * 2
    
    # First assignment: a[j] = b[i] + d[i] * e[i]
    first_val = b_val + d_val * e_val
    tl.store(a_ptr + j_base, first_val)
    
    # Conditional second assignment
    if c_val > 0.0:
        second_val = c_val + d_val * e_val
        tl.store(a_ptr + j_base + 1, second_val)

def s123_triton(a, b, c, d, e):
    n_half = b.shape[0] // 2
    
    # Launch one thread per i value
    grid = (n_half,)
    BLOCK_SIZE = 1
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_half,
        BLOCK_SIZE=BLOCK_SIZE,
    )