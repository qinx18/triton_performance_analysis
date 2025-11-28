import torch
import triton
import triton.language as tl

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the entire array sequentially since j depends on all previous iterations
    # We need to handle the dependent accumulation of j
    
    # Load all elements we'll need
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Process elements sequentially in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        block_offsets = block_start + offsets
        block_mask = block_offsets < n_elements
        
        # Load input arrays for this block
        b_vals = tl.load(b_ptr + block_offsets, mask=block_mask, other=0.0)
        c_vals = tl.load(c_ptr + block_offsets, mask=block_mask, other=0.0)
        d_vals = tl.load(d_ptr + block_offsets, mask=block_mask, other=0.0)
        e_vals = tl.load(e_ptr + block_offsets, mask=block_mask, other=0.0)
        
        # Process each element in the block sequentially
        for i in range(BLOCK_SIZE):
            if block_start + i >= n_elements:
                break
                
            # Calculate j = block_start + i (since j increments by 1 each iteration)
            j = block_start + i
            
            # Compute the result based on condition
            if b_vals[i] > 0.0:
                result = b_vals[i] + d_vals[i] * e_vals[i]
            else:
                result = c_vals[i] + d_vals[i] * e_vals[i]
            
            # Store result at position j
            tl.store(a_ptr + j, result)

def s124_triton(a, b, c, d, e):
    n_elements = b.shape[0]
    
    # Launch kernel with single block since we need sequential processing
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_elements))
    
    # Since this has sequential dependencies, we use a single thread block approach
    s124_kernel[(1,)](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a