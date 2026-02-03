import triton
import triton.language as tl

@triton.jit
def s1221_kernel(a_ptr, b_ptr, n_elements, STRIP_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for strip_start in range(0, n_elements, STRIP_SIZE):
        idx = strip_start + offsets
        mask = (idx < n_elements) & (idx >= 0)
        
        a_vals = tl.load(a_ptr + idx, mask=mask)
        b_read_vals = tl.load(b_ptr + idx - 4, mask=mask)
        
        result = b_read_vals + a_vals
        
        tl.store(b_ptr + idx, result, mask=mask)

def s1221_triton(a, b):
    N = a.shape[0]
    n_elements = N - 4
    
    STRIP_SIZE = 4
    BLOCK_SIZE = 4
    
    s1221_kernel[(1,)](
        a[4:],
        b[4:],
        n_elements,
        STRIP_SIZE=STRIP_SIZE,
        BLOCK_SIZE=BLOCK_SIZE
    )