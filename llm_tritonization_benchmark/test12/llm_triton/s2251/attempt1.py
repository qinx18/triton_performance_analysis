import torch
import triton
import triton.language as tl

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must be executed with a single block due to the sequential dependency
    # Each element depends on the previous computation of s
    
    offsets = tl.arange(0, BLOCK_SIZE)
    s = 0.0
    
    # Process elements in sequential blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load current elements
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block sequentially
        for i in range(BLOCK_SIZE):
            if block_start + i >= n_elements:
                break
                
            # Extract scalar values for sequential processing
            e_val = tl.load(e_ptr + block_start + i)
            b_val = tl.load(b_ptr + block_start + i)
            c_val = tl.load(c_ptr + block_start + i)
            d_val = tl.load(d_ptr + block_start + i)
            
            # a[i] = s * e[i]
            a_val = s * e_val
            tl.store(a_ptr + block_start + i, a_val)
            
            # s = b[i] + c[i]
            s = b_val + c_val
            
            # b[i] = a[i] + d[i]
            b_val = a_val + d_val
            tl.store(b_ptr + block_start + i, b_val)

def s2251_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Use single block execution due to sequential dependencies
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s2251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )