import triton
import triton.language as tl

@triton.jit
def s211_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel has loop-carried dependencies and must be processed sequentially
    # Only use one thread to process the entire range
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process sequentially from i=1 to n-2 (since original loop is i=1 to n-1-1)
    for i in range(1, n - 1):
        # Load values for current iteration
        b_prev = tl.load(b_ptr + i - 1)
        b_next = tl.load(b_ptr + i + 1)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # Compute and store results
        a_val = b_prev + c_val * d_val
        b_val = b_next - e_val * d_val
        
        tl.store(a_ptr + i, a_val)
        tl.store(b_ptr + i, b_val)

def s211_triton(a, b, c, d, e):
    n = a.shape[0]
    
    # Use only one thread block with one thread since we need sequential processing
    grid = (1,)
    BLOCK_SIZE = 1
    
    s211_kernel[grid](
        a, b, c, d, e, n,
        BLOCK_SIZE=BLOCK_SIZE
    )