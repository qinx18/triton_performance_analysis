import torch
import triton
import triton.language as tl

@triton.jit
def s1113_kernel(
    a_ptr,
    a_copy_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    # Create mask for valid elements
    mask = idx < n_elements
    
    # Load the broadcast value a[LEN_1D/2] (same for all threads)
    mid_idx = n_elements // 2
    a_mid = tl.load(a_copy_ptr + mid_idx)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + idx, mask=mask)
    
    # Compute a[i] = a[LEN_1D/2] + b[i]
    result = a_mid + b_vals
    
    # Store results
    tl.store(a_ptr + idx, result, mask=mask)

def s1113_triton(a, b):
    n_elements = a.shape[0]
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s1113_kernel[grid](
        a,
        a_copy,
        b,
        n_elements,
        BLOCK_SIZE,
    )