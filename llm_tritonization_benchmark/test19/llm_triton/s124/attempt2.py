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
    block_start = 0
    while block_start < n_elements:
        current_block_size = min(BLOCK_SIZE, n_elements - block_start)
        
        # Process each element in the current block sequentially
        idx = 0
        while idx < current_block_size:
            if block_start + idx >= n_elements:
                return
                
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
            
            idx = idx + 1
        
        block_start = block_start + BLOCK_SIZE

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