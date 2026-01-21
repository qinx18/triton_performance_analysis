import triton
import triton.language as tl

@triton.jit
def s221_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This is a strictly sequential pattern - use single thread
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process all elements sequentially starting from index 1
    for i in range(1, n):
        # Load c[i] and d[i]
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        # Load a[i] and update it
        a_val = tl.load(a_ptr + i)
        new_a_val = a_val + c_val * d_val
        tl.store(a_ptr + i, new_a_val)
        
        # Load b[i-1] and b[i], then update b[i]
        b_prev = tl.load(b_ptr + i - 1)
        b_val = tl.load(b_ptr + i)
        new_b_val = b_prev + new_a_val + d_val
        tl.store(b_ptr + i, new_b_val)

def s221_triton(a, b, c, d):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Use single thread since this is strictly sequential
    grid = (1,)
    
    s221_kernel[grid](
        a, b, c, d,
        n, BLOCK_SIZE
    )