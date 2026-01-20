import triton
import triton.language as tl
import torch

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the loop sequentially since j depends on previous iterations
    # We'll use a single thread to maintain the sequential dependency
    
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    j = -1
    
    # Process elements sequentially in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE)
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load data for this block
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block sequentially
        for idx in range(BLOCK_SIZE):
            if block_start + idx >= n_elements:
                break
                
            # Extract scalar values
            b_val = tl.load(b_ptr + block_start + idx)
            c_val = tl.load(c_ptr + block_start + idx)
            d_val = tl.load(d_ptr + block_start + idx)
            e_val = tl.load(e_ptr + block_start + idx)
            
            j = j + 1
            
            # Compute based on condition
            if b_val > 0.0:
                result = b_val + d_val * e_val
            else:
                result = c_val + d_val * e_val
                
            # Store result
            tl.store(a_ptr + j, result)

def s124_triton(a, b, c, d, e):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    
    # Launch with single program to maintain sequential execution
    grid = (1,)
    
    s124_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a