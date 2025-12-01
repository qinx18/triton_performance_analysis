import triton
import triton.language as tl
import torch

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must run with a single thread block due to sequential dependencies
    pid = tl.program_id(axis=0)
    if pid != 0:
        return
    
    offsets = tl.arange(0, BLOCK_SIZE)
    s = 0.0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load arrays
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element sequentially within the block
        for i in range(BLOCK_SIZE):
            if block_start + i >= n_elements:
                break
                
            # a[i] = s * e[i]
            a_val = s * e_vals[i]
            
            # s = b[i] + c[i]
            s = b_vals[i] + c_vals[i]
            
            # b[i] = a[i] + d[i]
            b_new_val = a_val + d_vals[i]
            
            # Store results
            tl.store(a_ptr + block_start + i, a_val)
            tl.store(b_ptr + block_start + i, b_new_val)

def s2251_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    grid = lambda meta: (1,)  # Single thread block due to dependencies
    
    s2251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )