import torch
import triton
import triton.language as tl

@triton.jit
def s112_kernel(
    a_ptr,
    a_copy_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load from read-only copy and b array
    a_vals = tl.load(a_copy_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute: a[i+1] = a[i] + b[i]
    # Store to i+1 positions in original array
    store_indices = indices + 1
    store_mask = mask & (store_indices < n_elements)
    
    result = a_vals + b_vals
    tl.store(a_ptr + store_indices, result, mask=store_mask)

def s112_triton(a, b):
    n_elements = a.shape[0] - 1  # We process indices 0 to LEN_1D-2
    
    # Create read-only copy to avoid WAR race condition
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s112_kernel[grid](
        a,
        a_copy,
        b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )