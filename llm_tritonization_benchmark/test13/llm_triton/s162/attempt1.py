import torch
import triton
import triton.language as tl

@triton.jit
def s162_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements, k,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and calculate offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Create mask for valid indices
    mask = indices < n_elements
    
    # Load b[i] and c[i]
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    
    # Load a[i + k] with bounds checking
    read_indices = indices + k
    read_mask = mask & (read_indices < (n_elements + k))
    a_read_vals = tl.load(a_ptr + read_indices, mask=read_mask)
    
    # Compute: a[i] = a[i + k] + b[i] * c[i]
    result = a_read_vals + b_vals * c_vals
    
    # Store result back to a[i]
    tl.store(a_ptr + indices, result, mask=mask)

def s162_triton(a, b, c, k):
    if k <= 0:
        return
    
    # Calculate number of elements to process (LEN_1D - 1)
    n_elements = a.shape[0] - 1
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s162_kernel[grid](
        a, b, c,
        n_elements, k,
        BLOCK_SIZE=BLOCK_SIZE,
    )