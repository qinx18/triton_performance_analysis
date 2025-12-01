import triton
import triton.language as tl
import torch

@triton.jit
def s121_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Read-only copy to avoid WAR dependency
    a_temp = tl.zeros([n_elements], dtype=tl.float32)
    
    # Load entire array a into temporary storage
    offsets = tl.arange(0, BLOCK_SIZE)
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        tl.store(a_temp + current_offsets, a_vals, mask=mask)
    
    # Perform computation: a[i] = a[i+1] + b[i]
    for block_start in range(0, n_elements - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < (n_elements - 1)
        
        # Load a[i+1] and b[i]
        a_next = tl.load(a_temp + current_offsets + 1, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute and store result
        result = a_next + b_vals
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s121_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel with single block
    s121_kernel[(1,)](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a