import torch
import triton
import triton.language as tl

@triton.jit
def s123_kernel(a, b, c, d, e, n):
    offsets = tl.arange(0, 1)
    
    j = -1
    for i in range(n):
        j += 1
        
        # Load values for current i
        i_offset = i
        mask_i = i_offset < n
        
        b_val = tl.load(b + i_offset, mask=mask_i)
        d_val = tl.load(d + i_offset, mask=mask_i)
        e_val = tl.load(e + i_offset, mask=mask_i)
        c_val = tl.load(c + i_offset, mask=mask_i)
        
        # First assignment: a[j] = b[i] + d[i] * e[i]
        result1 = b_val + d_val * e_val
        j_offset = j
        mask_j = j_offset >= 0
        tl.store(a + j_offset, result1, mask=mask_j)
        
        # Conditional assignment
        if c_val > 0.0:
            j += 1
            result2 = c_val + d_val * e_val
            j_offset = j
            mask_j = j_offset >= 0
            tl.store(a + j_offset, result2, mask=mask_j)

def s123_triton(a, b, c, d, e):
    n = b.shape[0] // 2
    
    # Launch single thread since j is data-dependent
    grid = (1,)
    s123_kernel[grid](a, b, c, d, e, n)
    
    return a