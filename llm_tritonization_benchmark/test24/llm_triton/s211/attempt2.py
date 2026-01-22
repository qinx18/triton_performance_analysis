import triton
import triton.language as tl
import torch

@triton.jit
def s211_kernel(a_ptr, b_ptr, b_copy_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This is strictly sequential due to RAW dependency
    pid = tl.program_id(0)
    
    # Only use one block to ensure sequential execution
    if pid != 0:
        return
    
    # Process sequentially from 1 to n-2 (inclusive)
    for i in range(1, n - 1):
        # Load values for a[i] = b[i-1] + c[i] * d[i]
        b_prev = tl.load(b_ptr + (i - 1))
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        # Compute and store a[i]
        a_val = b_prev + c_val * d_val
        tl.store(a_ptr + i, a_val)
        
        # Load values for b[i] = b[i+1] - e[i] * d[i]
        b_next = tl.load(b_copy_ptr + (i + 1))
        e_val = tl.load(e_ptr + i)
        
        # Compute and store b[i]
        b_val = b_next - e_val * d_val
        tl.store(b_ptr + i, b_val)

def s211_triton(a, b, c, d, e):
    n = a.shape[0]
    
    # Create read-only copy for WAR safety
    b_copy = b.clone()
    
    # Use single block for sequential execution
    BLOCK_SIZE = 1
    grid = (1,)
    
    # Launch kernel
    s211_kernel[grid](
        a, b, b_copy, c, d, e, n,
        BLOCK_SIZE=BLOCK_SIZE
    )