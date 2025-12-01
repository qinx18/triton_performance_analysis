import triton
import triton.language as tl
import torch

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel cannot be parallelized due to loop-carried dependencies
    # Only one thread block should execute this
    pid = tl.program_id(axis=0)
    
    if pid != 0:
        return
    
    s = 0.0
    
    for i in range(0, n_elements, BLOCK_SIZE):
        mask = i + tl.arange(0, BLOCK_SIZE) < n_elements
        offsets = i + tl.arange(0, BLOCK_SIZE)
        
        # Load data
        b_vals = tl.load(b_ptr + offsets, mask=mask)
        c_vals = tl.load(c_ptr + offsets, mask=mask)
        d_vals = tl.load(d_ptr + offsets, mask=mask)
        e_vals = tl.load(e_ptr + offsets, mask=mask)
        
        # Process each element sequentially
        for j in range(BLOCK_SIZE):
            if i + j < n_elements:
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
    
    grid = (1,)  # Only one block to maintain sequential execution
    
    s2251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )