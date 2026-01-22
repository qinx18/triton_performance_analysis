import triton
import triton.language as tl

@triton.jit
def s1112_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        # Reverse indexing: start from the end
        idx = n - 1 - block_start - offsets
        mask = (idx >= 0) & (idx < n)
        
        b_vals = tl.load(b_ptr + idx, mask=mask)
        result = b_vals + 1.0
        tl.store(a_ptr + idx, result, mask=mask)

def s1112_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s1112_kernel[grid](a, b, n, BLOCK_SIZE=BLOCK_SIZE)