import triton
import triton.language as tl

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Sequential execution required due to dependencies
    for i in range(n - 1):
        # Load values for current iteration
        a_i = tl.load(a_ptr + i)
        b_i = tl.load(b_ptr + i)
        c_i = tl.load(c_ptr + i)
        d_i = tl.load(d_ptr + i)
        a_i_plus_1 = tl.load(a_ptr + i + 1)
        
        # Execute statements in order
        # S0: a[i] = b[i] + c[i] * d[i]
        new_a_i = b_i + c_i * d_i
        tl.store(a_ptr + i, new_a_i)
        
        # S1: b[i] = c[i] + b[i]
        new_b_i = c_i + b_i
        tl.store(b_ptr + i, new_b_i)
        
        # S2: a[i+1] = b[i] + a[i+1] * d[i] (using updated b[i])
        new_a_i_plus_1 = new_b_i + a_i_plus_1 * d_i
        tl.store(a_ptr + i + 1, new_a_i_plus_1)

def s244_triton(a, b, c, d):
    n = a.shape[0]
    
    # Use single thread since we need sequential execution
    grid = (1,)
    s244_kernel[grid](a, b, c, d, n, BLOCK_SIZE=1)