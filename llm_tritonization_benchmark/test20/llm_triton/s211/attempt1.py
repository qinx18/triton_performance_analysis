import triton
import triton.language as tl

@triton.jit
def s211_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel is strictly sequential due to loop-carried dependency
    # Only use one thread to process all elements sequentially
    thread_id = tl.program_id(0)
    
    if thread_id != 0:
        return
    
    # Process elements sequentially from index 1 to n-2
    for i in range(1, n - 1):
        # Load values for current iteration
        b_prev = tl.load(b_ptr + i - 1)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        b_next = tl.load(b_ptr + i + 1)
        e_val = tl.load(e_ptr + i)
        
        # Compute and store a[i]
        a_val = b_prev + c_val * d_val
        tl.store(a_ptr + i, a_val)
        
        # Compute and store b[i]
        b_val = b_next - e_val * d_val
        tl.store(b_ptr + i, b_val)

def s211_triton(a, b, c, d, e):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch with single block since computation is strictly sequential
    grid = (1,)
    
    s211_kernel[grid](
        a, b, c, d, e, n,
        BLOCK_SIZE=BLOCK_SIZE
    )