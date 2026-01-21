import triton
import triton.language as tl
import torch

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the entire array sequentially to maintain the j increment logic
    # We use a single thread block to preserve the sequential nature of the induction variable
    
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Initialize j to -1 (will be incremented before each use)
    j = -1
    
    # Process elements in blocks while maintaining sequential order
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load input arrays
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block sequentially
        for i in range(BLOCK_SIZE):
            current_idx = block_start + i
            if current_idx >= n_elements:
                continue
                
            # Increment j for each element
            j += 1
            
            # Select value based on condition
            if b_vals[i] > 0.0:
                result = b_vals[i] + d_vals[i] * e_vals[i]
            else:
                result = c_vals[i] + d_vals[i] * e_vals[i]
            
            # Store result at position j
            tl.store(a_ptr + j, result)

def s124_triton(a, b, c, d, e):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Use single grid since we need sequential processing
    grid = (1,)
    
    s124_kernel[grid](
        a, b, c, d, e,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )