import triton
import triton.language as tl

@triton.jit
def s175_kernel(a_ptr, a_copy_ptr, b_ptr, inc, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE * inc):
        i_offsets = block_start + offsets * inc
        mask = i_offsets < n
        
        a_vals = tl.load(a_copy_ptr + i_offsets + inc, mask=mask)
        b_vals = tl.load(b_ptr + i_offsets, mask=mask)
        
        result = a_vals + b_vals
        
        tl.store(a_ptr + i_offsets, result, mask=mask)

def s175_triton(a, b, inc):
    n = a.shape[0] - 1
    
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s175_kernel[grid](a, a_copy, b, inc, n, BLOCK_SIZE=BLOCK_SIZE)