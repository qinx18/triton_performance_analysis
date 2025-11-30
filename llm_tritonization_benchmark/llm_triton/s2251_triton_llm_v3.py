import torch
import triton
import triton.language as tl

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must run with only one program since there's a sequential dependency
    pid = tl.program_id(axis=0)
    
    # Only the first program should execute
    if pid != 0:
        return
    
    s = 0.0
    
    # Process elements sequentially due to dependency
    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load current values
        e_vals = tl.load(e_ptr + offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
        
        # Process each element in the block sequentially
        for j in range(BLOCK_SIZE):
            if i + j >= n_elements:
                break
                
            # a[i] = s * e[i]
            a_val = s * e_vals[j]
            tl.store(a_ptr + i + j, a_val)
            
            # s = b[i] + c[i]
            s = b_vals[j] + c_vals[j]
            
            # b[i] = a[i] + d[i]
            b_val = a_val + d_vals[j]
            tl.store(b_ptr + i + j, b_val)

def s2251_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    
    # Launch with only one block since computation is sequential
    grid = (1,)
    
    s2251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )