import triton
import triton.language as tl
import torch

@triton.jit
def s277_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel must be strictly sequential due to b[i+1] = ... b[i] dependency
    # Use single thread to process all elements sequentially
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process elements sequentially from 0 to n-2 (since we access i+1)
    for i in range(n - 1):
        # Load a[i]
        a_val = tl.load(a_ptr + i)
        
        # Check if a[i] >= 0 - if so, goto L20 (skip everything)
        skip_all = a_val >= 0.0
        
        # Load b[i] only if we didn't skip
        b_val = tl.load(b_ptr + i)
        
        # Check if b[i] >= 0 - if so, goto L30 (skip a[i] update)
        skip_a_update = b_val >= 0.0
        
        # Execute a[i] += c[i] * d[i] only if both conditions are false
        should_update_a = (skip_all == 0) & (skip_a_update == 0)
        if should_update_a:
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            a_val += c_val * d_val
            tl.store(a_ptr + i, a_val)
        
        # L30: Execute b[i+1] = c[i] + d[i] * e[i] (unless we skipped everything)
        if skip_all == 0:
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            e_val = tl.load(e_ptr + i)
            result = c_val + d_val * e_val
            tl.store(b_ptr + i + 1, result)

def s277_triton(a, b, c, d, e):
    n = a.shape[0]
    
    # Use single thread since computation must be strictly sequential
    grid = (1,)
    BLOCK_SIZE = 1
    
    s277_kernel[grid](
        a, b, c, d, e,
        n,
        BLOCK_SIZE=BLOCK_SIZE
    )