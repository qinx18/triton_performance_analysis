import torch
import triton
import triton.language as tl

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Since we have dependencies between iterations, we need to process sequentially
    # Each block processes one chunk of the array sequentially
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Process each element in this block sequentially
    for local_i in range(BLOCK_SIZE):
        i = block_start + local_i
        if i >= n_elements:
            break
            
        # Load individual elements
        b_i = tl.load(b_ptr + i)
        c_i = tl.load(c_ptr + i)
        d_i = tl.load(d_ptr + i)
        a_i_plus_1 = tl.load(a_ptr + i + 1, mask=(i + 1 < n_elements + 1))
        
        # First operation: a[i] = b[i] + c[i] * d[i]
        a_new = b_i + c_i * d_i
        tl.store(a_ptr + i, a_new)
        
        # Second operation: b[i] = c[i] + b[i]
        b_new = c_i + b_i
        tl.store(b_ptr + i, b_new)
        
        # Third operation: a[i+1] = b[i] + a[i+1] * d[i]
        # Note: b[i] is now the updated value from second operation
        if i + 1 < n_elements + 1:
            a_i_plus_1_new = b_new + a_i_plus_1 * d_i
            tl.store(a_ptr + i + 1, a_i_plus_1_new)

def s244_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    # Use small block size to handle dependencies
    BLOCK_SIZE = 32
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s244_kernel[grid](
        a, b, c, d, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, b