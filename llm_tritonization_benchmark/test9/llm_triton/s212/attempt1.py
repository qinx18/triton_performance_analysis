import torch
import triton
import triton.language as tl

@triton.jit
def s212_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load a[i] and c[i]
    a_vals = tl.load(a_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    # First statement: a[i] *= c[i]
    a_new = a_vals * c_vals
    tl.store(a_ptr + idx, a_new, mask=mask)
    
    # Load a[i+1] and d[i] for second statement
    idx_plus_1 = idx + 1
    mask_plus_1 = idx_plus_1 < (n_elements + 1)  # Allow loading a[i+1] where i+1 can be up to n_elements
    a_plus_1_vals = tl.load(a_ptr + idx_plus_1, mask=mask_plus_1)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # Load current b[i]
    b_vals = tl.load(b_ptr + idx, mask=mask)
    
    # Second statement: b[i] += a[i + 1] * d[i]
    # Use updated a values for a[i+1]
    b_new = b_vals + a_plus_1_vals * d_vals
    tl.store(b_ptr + idx, b_new, mask=mask)

def s212_triton(a, b, c, d):
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 256
    
    # Create temporary copy of a to handle WAR dependency
    a_temp = a.clone()
    
    # Launch kernel with sequential processing to handle dependency
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_size = min(BLOCK_SIZE, n_elements - block_start)
        grid = (1,)
        
        s212_kernel[grid](
            a_temp[block_start:], b[block_start:], c[block_start:], d[block_start:],
            current_size, BLOCK_SIZE
        )
    
    # Copy result back to original array
    a[:n_elements] = a_temp[:n_elements]