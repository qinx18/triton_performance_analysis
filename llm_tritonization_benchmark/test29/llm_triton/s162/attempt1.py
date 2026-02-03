import triton
import triton.language as tl
import torch

@triton.jit
def s162_kernel(
    a_ptr, a_copy_ptr, b_ptr, c_ptr,
    n, k,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    # Create mask for valid elements (i < n-1)
    mask = current_offsets < (n - 1)
    
    # Load from arrays
    a_copy_vals = tl.load(a_copy_ptr + current_offsets + k, mask=mask)
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    
    # Compute: a[i] = a[i + k] + b[i] * c[i]
    result = a_copy_vals + b_vals * c_vals
    
    # Store result
    tl.store(a_ptr + current_offsets, result, mask=mask)

def s162_triton(a, b, c, k):
    if k <= 0:
        return
    
    n = a.shape[0]
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    # Calculate grid size
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n - 1, BLOCK_SIZE)
    
    # Launch kernel
    s162_kernel[grid_size](
        a, a_copy, b, c,
        n, k,
        BLOCK_SIZE=BLOCK_SIZE
    )