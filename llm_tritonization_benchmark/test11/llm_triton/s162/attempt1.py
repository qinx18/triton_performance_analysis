import torch
import triton
import triton.language as tl

@triton.jit
def s162_kernel(a_ptr, b_ptr, c_ptr, n_elements, k, BLOCK_SIZE: tl.constexpr):
    # Get block of indices
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    # Mask for valid indices
    mask = idx < n_elements
    
    # Load b[i] and c[i]
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    # Load a[i + k] with bounds checking
    read_idx = idx + k
    read_mask = mask & (read_idx < (n_elements + k))
    a_vals = tl.load(a_ptr + read_idx, mask=read_mask)
    
    # Compute a[i] = a[i + k] + b[i] * c[i]
    result = a_vals + b_vals * c_vals
    
    # Store result back to a[i]
    tl.store(a_ptr + idx, result, mask=mask)

def s162_triton(a, b, c, k):
    if k <= 0:
        return
    
    # Work on LEN_1D-1 elements as per original code
    n_elements = len(a) - 1
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch kernel
    s162_kernel[(grid_size,)](
        a, b, c, n_elements, k, BLOCK_SIZE
    )