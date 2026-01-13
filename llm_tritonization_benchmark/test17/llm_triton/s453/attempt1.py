import torch
import triton
import triton.language as tl

@triton.jit
def s453_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must be launched with num_programs=1 to maintain sequential dependency
    pid = tl.program_id(0)
    
    offsets = tl.arange(0, BLOCK_SIZE)
    s = 0.0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load b values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Update s for each element in the block
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                s += 2.0
                # Store s * b[i] to a[i]
                result = s * tl.load(b_ptr + block_start + i)
                tl.store(a_ptr + block_start + i, result)

def s453_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch with single program to maintain sequential dependency
    grid = (1,)
    
    s453_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a