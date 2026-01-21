import triton
import triton.language as tl

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, N: tl.constexpr):
    # This kernel runs with a single thread since the computation is strictly sequential
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    # Process sequentially from i=2 to N-1
    for i in range(2, N):
        # Load current values
        a_i = tl.load(a_ptr + i)
        a_i_minus_1 = tl.load(a_ptr + i - 1)
        a_i_minus_2 = tl.load(a_ptr + i - 2)
        b_i = tl.load(b_ptr + i)
        c_i = tl.load(c_ptr + i)
        
        # Compute: a[i] = a[i] + a[i - 1] * b[i] + a[i - 2] * c[i]
        result = a_i + a_i_minus_1 * b_i + a_i_minus_2 * c_i
        
        # Store result
        tl.store(a_ptr + i, result)

def s322_triton(a, b, c):
    N = a.shape[0]
    
    # Use single thread grid since computation must be sequential
    grid = (1,)
    
    s322_kernel[grid](
        a, b, c, N
    )