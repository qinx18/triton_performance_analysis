import torch
import triton
import triton.language as tl

@triton.jit
def s342_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    
    # Process sequentially since j depends on previous iterations
    if block_id == 0:
        j = -1
        offsets = tl.arange(0, BLOCK_SIZE)
        
        for block_start in range(0, n_elements, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < n_elements
            
            # Load current block of a
            a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
            
            # Process each element in the block
            for i in range(BLOCK_SIZE):
                if block_start + i >= n_elements:
                    break
                    
                # Check if a[i] > 0
                if block_start + i < n_elements:
                    a_val = tl.load(a_ptr + block_start + i)
                    if a_val > 0.0:
                        j += 1
                        if j < n_elements:
                            b_val = tl.load(b_ptr + j)
                            tl.store(a_ptr + block_start + i, b_val)

def s342_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)  # Single block since we need sequential processing
    
    s342_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )