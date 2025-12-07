import torch
import triton
import triton.language as tl

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the conditional computation sequentially
    # due to the dependency on induction variable j
    
    offsets = tl.arange(0, BLOCK_SIZE)
    j = -1
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load input arrays
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block
        for i in range(BLOCK_SIZE):
            if block_start + i >= n_elements:
                break
                
            j += 1
            
            # Compute based on condition
            if b_vals[i] > 0.0:
                result = b_vals[i] + d_vals[i] * e_vals[i]
            else:
                result = c_vals[i] + d_vals[i] * e_vals[i]
            
            # Store result at position j
            tl.store(a_ptr + j, result)

def s124_triton(a, b, c, d, e):
    n_elements = b.numel()
    
    # Use small block size since we need sequential processing
    BLOCK_SIZE = 32
    
    # Launch kernel with single program
    grid = (1,)
    s124_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a