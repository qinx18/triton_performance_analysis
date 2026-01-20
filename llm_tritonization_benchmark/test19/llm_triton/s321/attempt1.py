import triton
import triton.language as tl

@triton.jit
def s321_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process sequentially since each element depends on the previous one
    for i in range(1, n):
        if i < n:
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
    BLOCK_SIZE = 256
    
    # Launch kernel with single thread since this is sequential
    grid = (1,)
    s321_kernel[grid](a, b, n, BLOCK_SIZE=BLOCK_SIZE)