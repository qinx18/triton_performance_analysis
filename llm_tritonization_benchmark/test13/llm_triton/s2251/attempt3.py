import torch
import triton
import triton.language as tl

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Sequential processing - each thread block processes the entire array
    block_id = tl.program_id(0)
    if block_id > 0:
        return
    
    s = 0.0
    
    # Process array sequentially in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE)
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load arrays
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the current block
        num_elements = min(BLOCK_SIZE, n_elements - block_start)
        for i in range(num_elements):
            # a[i] = s * e[i]
            e_val = tl.load(e_ptr + block_start + i)
            a_val = s * e_val
            tl.store(a_ptr + block_start + i, a_val)
            
            # s = b[i] + c[i]
            b_val = tl.load(b_ptr + block_start + i)
            c_val = tl.load(c_ptr + block_start + i)
            s = b_val + c_val
            
            # b[i] = a[i] + d[i]
            d_val = tl.load(d_ptr + block_start + i)
            new_b_val = a_val + d_val
            tl.store(b_ptr + block_start + i, new_b_val)

def s2251_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Launch kernel with single thread block for sequential processing
    grid = (1,)
    s2251_kernel[grid](a, b, c, d, e, n_elements, BLOCK_SIZE)