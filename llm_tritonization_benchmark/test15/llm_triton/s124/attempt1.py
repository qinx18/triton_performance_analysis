import torch
import triton
import triton.language as tl

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize output index counter
    j = 0
    
    # Process elements in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load input arrays
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute d[i] * e[i] for both branches
        de_product = d_vals * e_vals
        
        # Process each element in the block
        for i in range(BLOCK_SIZE):
            if block_start + i >= n_elements:
                break
                
            element_mask = tl.full([1], block_start + i < n_elements, dtype=tl.int1)
            
            if element_mask:
                b_val = tl.load(b_ptr + block_start + i)
                c_val = tl.load(c_ptr + block_start + i)
                d_val = tl.load(d_ptr + block_start + i)
                e_val = tl.load(e_ptr + block_start + i)
                
                de_val = d_val * e_val
                
                if b_val > 0.0:
                    result = b_val + de_val
                else:
                    result = c_val + de_val
                
                tl.store(a_ptr + j, result)
                j += 1

def s124_triton(a, b, c, d, e):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel with single block since we need sequential j indexing
    s124_kernel[(1,)](
        a, b, c, d, e, 
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )