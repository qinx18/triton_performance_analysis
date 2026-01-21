import triton
import triton.language as tl

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Sequential execution due to dependencies
    for i in range(N - 1):
        # Load values for current iteration
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        a_next_val = tl.load(a_ptr + i + 1)
        
        # Statement 1: a[i] = b[i] + c[i] * d[i]
        a_val = b_val + c_val * d_val
        tl.store(a_ptr + i, a_val)
        
        # Statement 2: b[i] = c[i] + b[i]
        b_new = c_val + b_val
        tl.store(b_ptr + i, b_new)
        
        # Statement 3: a[i+1] = b[i] + a[i+1] * d[i]
        # Use the updated b[i] value
        a_next_new = b_new + a_next_val * d_val
        tl.store(a_ptr + i + 1, a_next_new)

def s244_triton(a, b, c, d):
    N = a.shape[0]
    
    # Use single-threaded execution due to complex dependencies
    grid = (1,)
    s244_kernel[grid](a, b, c, d, N, BLOCK_SIZE=1)