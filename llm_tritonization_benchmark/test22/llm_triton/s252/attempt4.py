import triton
import triton.language as tl

@triton.jit
def s252_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This is a sequential scan operation that cannot be parallelized
    # Only process if this is thread 0
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Initialize t = 0
    t = 0.0
    
    # Process elements sequentially
    for i in range(n):
        # Load b and c values
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        
        # s = b[i] * c[i]
        s = b_val * c_val
        
        # a[i] = s + t
        a_val = s + t
        tl.store(a_ptr + i, a_val)
        
        # t = s
        t = s

def s252_triton(a, b, c):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Since this is sequential, we only need one thread block
    grid = (1,)
    
    s252_kernel[grid](
        a, b, c, n,
        BLOCK_SIZE=BLOCK_SIZE
    )