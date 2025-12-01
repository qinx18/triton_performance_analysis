import triton
import triton.language as tl
import torch

@triton.jit
def s254_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:
        # First block handles the sequential computation
        offsets = tl.arange(0, BLOCK_SIZE)
        
        # Load x = b[n_elements-1]
        x_offset = n_elements - 1
        x = tl.load(b_ptr + x_offset)
        
        # Process elements sequentially in blocks
        for block_start in range(0, n_elements, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < n_elements
            
            # Load b values for this block
            b_vals = tl.load(b_ptr + current_offsets, mask=mask)
            
            # Compute a[i] = (b[i] + x) * 0.5 and update x sequentially
            for i in range(BLOCK_SIZE):
                if block_start + i < n_elements:
                    b_val = tl.load(b_ptr + block_start + i)
                    a_val = (b_val + x) * 0.5
                    tl.store(a_ptr + block_start + i, a_val)
                    x = b_val

def s254_triton(a, b):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (1,)  # Only one block to handle sequential dependency
    
    s254_kernel[grid](
        a, b, n_elements, BLOCK_SIZE
    )