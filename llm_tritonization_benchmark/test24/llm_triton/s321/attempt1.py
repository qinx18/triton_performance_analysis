import triton
import triton.language as tl

@triton.jit
def s321_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This is a first-order linear recurrence that cannot be parallelized
    # Must compute sequentially from i=1 to n-1
    for i in range(1, n):
        # Load a[i-1], a[i], and b[i]
        a_prev = tl.load(a_ptr + (i - 1))
        a_curr = tl.load(a_ptr + i)
        b_curr = tl.load(b_ptr + i)
        
        # Compute a[i] += a[i-1] * b[i]
        result = a_curr + a_prev * b_curr
        
        # Store result back to a[i]
        tl.store(a_ptr + i, result)

def s321_triton(a, b):
    n = a.shape[0]
    
    # Launch single thread since this is a sequential recurrence
    grid = (1,)
    BLOCK_SIZE = 1
    
    s321_kernel[grid](
        a, b, n, 
        BLOCK_SIZE=BLOCK_SIZE
    )