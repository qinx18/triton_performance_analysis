import torch
import triton
import triton.language as tl

@triton.jit
def s221_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    if pid == 0:
        # Handle i=1 separately
        a1 = tl.load(a_ptr + 1)
        c1 = tl.load(c_ptr + 1)
        d1 = tl.load(d_ptr + 1)
        b0 = tl.load(b_ptr + 0)
        
        # a[1] += c[1] * d[1]
        new_a1 = a1 + c1 * d1
        tl.store(a_ptr + 1, new_a1)
        
        # b[1] = b[0] + a[1] + d[1]
        new_b1 = b0 + new_a1 + d1
        tl.store(b_ptr + 1, new_b1)
        
        # Process remaining elements sequentially
        for i in range(2, n_elements):
            # Load values
            a_val = tl.load(a_ptr + i)
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            b_prev = tl.load(b_ptr + i - 1)
            
            # a[i] += c[i] * d[i]
            new_a = a_val + c_val * d_val
            tl.store(a_ptr + i, new_a)
            
            # b[i] = b[i-1] + a[i] + d[i]
            new_b = b_prev + new_a + d_val
            tl.store(b_ptr + i, new_b)

def s221_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    if n_elements <= 1:
        return
    
    BLOCK_SIZE = 1
    grid = (1,)
    
    s221_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )