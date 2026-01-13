import torch
import triton
import triton.language as tl

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel cannot be parallelized due to sequential dependency on scalar s
    # We need to process all elements sequentially in a single thread block
    
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process elements sequentially
    s = 0.0
    
    # Process in blocks but sequentially
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load current block
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block sequentially
        for i in range(BLOCK_SIZE):
            if block_start + i >= n_elements:
                break
                
            # a[i] = s * e[i]
            a_val = s * e_vals[i] if mask[i] else 0.0
            
            # s = b[i] + c[i]
            if mask[i]:
                s = b_vals[i] + c_vals[i]
            
            # b[i] = a[i] + d[i]
            if mask[i]:
                b_new_val = a_val + d_vals[i]
                tl.store(a_ptr + block_start + i, a_val)
                tl.store(b_ptr + block_start + i, b_new_val)

def s2251_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Launch with single block since we need sequential processing
    grid = (1,)
    
    s2251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )