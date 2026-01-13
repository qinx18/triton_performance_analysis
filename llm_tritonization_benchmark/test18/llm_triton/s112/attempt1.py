import triton
import triton.language as tl
import torch

@triton.jit
def s112_kernel(a_ptr, a_copy_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load from read-only copy and b array
    a_vals = tl.load(a_copy_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    
    # Compute a[i] + b[i]
    result = a_vals + b_vals
    
    # Store to a[i+1] (offset by +1)
    store_idx = idx + 1
    store_mask = store_idx < (n_elements + 1)
    tl.store(a_ptr + store_idx, result, mask=store_mask)

def s112_triton(a, b):
    n_elements = a.shape[0] - 1  # Process indices 0 to LEN_1D-2
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s112_kernel[grid](
        a, a_copy, b, n_elements, BLOCK_SIZE
    )