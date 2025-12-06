import triton
import triton.language as tl
import torch

@triton.jit
def s1244_kernel(
    a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    # Mask for valid elements (we process n_elements-1 elements)
    mask = idx < (n_elements - 1)
    
    # Load values
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    # Compute a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i]
    a_vals = b_vals + c_vals * c_vals + b_vals * b_vals + c_vals
    
    # Store a[i]
    tl.store(a_ptr + idx, a_vals, mask=mask)
    
    # Load a[i+1] from the read-only copy for d[i] computation
    idx_plus1 = idx + 1
    mask_plus1 = idx_plus1 < n_elements
    a_plus1_vals = tl.load(a_copy_ptr + idx_plus1, mask=mask_plus1)
    
    # Compute d[i] = a[i] + a[i+1]
    # We need to mask this properly - only valid where both a[i] and a[i+1] exist
    final_mask = mask & mask_plus1
    d_vals = a_vals + a_plus1_vals
    
    # Store d[i]
    tl.store(d_ptr + idx, d_vals, mask=final_mask)

def s1244_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Create read-only copy of array a to handle WAR dependency
    a_copy = a.clone()
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size for processing n_elements-1 elements
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    # Launch kernel
    s1244_kernel[grid](
        a, a_copy, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )