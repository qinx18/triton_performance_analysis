import torch
import triton
import triton.language as tl

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must be executed with a single block since s has sequential dependency
    pid = tl.program_id(axis=0)
    if pid != 0:
        return
    
    s = 0.0
    
    # Process elements sequentially in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load current block
        e_vals = tl.load(e_ptr + offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
        
        # Process each element sequentially within the block
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                # a[i] = s * e[i]
                a_val = s * e_vals[i]
                
                # s = b[i] + c[i]
                s = b_vals[i] + c_vals[i]
                
                # b[i] = a[i] + d[i]
                b_new = a_val + d_vals[i]
                
                # Store results
                tl.store(a_ptr + block_start + i, a_val)
                tl.store(b_ptr + block_start + i, b_new)

def s2251_triton(a, b, c, d, e):
    n_elements = a.numel()
    
    # Use a single block to maintain sequential dependency
    BLOCK_SIZE = min(1024, n_elements)
    grid = (1,)
    
    s2251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )