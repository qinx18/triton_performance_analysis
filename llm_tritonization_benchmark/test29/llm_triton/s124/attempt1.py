import triton
import triton.language as tl
import torch

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize output index
    j = 0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load input values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute d[i] * e[i] once
        de_product = d_vals * e_vals
        
        # Process each element in the block sequentially
        for i in range(BLOCK_SIZE):
            if block_start + i >= n_elements:
                break
                
            # Extract scalar values for current element
            b_val = tl.load(b_ptr + block_start + i)
            c_val = tl.load(c_ptr + block_start + i)
            d_val = tl.load(d_ptr + block_start + i)
            e_val = tl.load(e_ptr + block_start + i)
            
            # Conditional computation
            if b_val > 0.0:
                result = b_val + d_val * e_val
            else:
                result = c_val + d_val * e_val
            
            # Store result at position j
            tl.store(a_ptr + j, result)
            j += 1

def s124_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s124_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )