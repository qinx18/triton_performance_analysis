import triton
import triton.language as tl

@triton.jit
def s211_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(1, N - 1):
        # a[i] = b[i - 1] + c[i] * d[i]
        b_prev = tl.load(b_ptr + (i - 1))
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        a_val = b_prev + c_val * d_val
        tl.store(a_ptr + i, a_val)
        
        # b[i] = b[i + 1] - e[i] * d[i]
        b_next = tl.load(b_ptr + (i + 1))
        e_val = tl.load(e_ptr + i)
        b_val = b_next - e_val * d_val
        tl.store(b_ptr + i, b_val)

def s211_triton(a, b, c, d, e):
    N = a.shape[0]
    
    BLOCK_SIZE = 1
    grid = (1,)
    
    s211_kernel[grid](
        a, b, c, d, e, N,
        BLOCK_SIZE=BLOCK_SIZE
    )