import triton
import triton.language as tl
import torch

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the entire array sequentially due to the dependency
    # We'll use a single thread block to maintain the sequential j increment
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    # Initialize j counter
    j = -1
    
    # Process elements sequentially
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load current block
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block
        for local_i in range(BLOCK_SIZE):
            global_i = block_start + local_i
            if global_i < n_elements:
                # Extract scalar values for this iteration
                b_val = tl.load(b_ptr + global_i)
                c_val = tl.load(c_ptr + global_i)
                d_val = tl.load(d_ptr + global_i)
                e_val = tl.load(e_ptr + global_i)
                
                # Always increment j
                j = j + 1
                
                # Conditional computation
                if b_val > 0.0:
                    result = b_val + d_val * e_val
                else:
                    result = c_val + d_val * e_val
                
                # Store result at position j
                tl.store(a_ptr + j, result)

def s124_triton(a, b, c, d, e):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    
    # Use single grid since we need sequential processing
    grid = (1,)
    
    s124_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )