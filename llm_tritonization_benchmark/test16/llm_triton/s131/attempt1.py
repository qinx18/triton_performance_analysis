import triton
import triton.language as tl
import torch

@triton.jit
def s131_kernel(a_ptr, a_copy_ptr, b_ptr, n_elements, m, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load from read-only copy for a[i + m] and from b for b[i]
    a_read_offsets = idx + m
    a_read_mask = a_read_offsets < (n_elements + 1)  # Original array size before truncation
    
    a_vals = tl.load(a_copy_ptr + a_read_offsets, mask=mask & a_read_mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    
    # Compute a[i] = a[i + m] + b[i]
    result = a_vals + b_vals
    
    # Store to original array
    tl.store(a_ptr + idx, result, mask=mask)

def s131_triton(a, b, m):
    n_elements = a.shape[0] - 1  # Process LEN_1D - 1 elements
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s131_kernel[grid](
        a, a_copy, b,
        n_elements, m,
        BLOCK_SIZE=BLOCK_SIZE
    )