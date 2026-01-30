import triton
import triton.language as tl
import torch

@triton.jit
def s112_kernel(
    a_ptr,
    a_copy_ptr,
    b_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < (n - 1)
        
        # Load from read-only copy and b array
        a_vals = tl.load(a_copy_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        # Compute result
        result = a_vals + b_vals
        
        # Store to original array at offset +1
        store_offsets = current_offsets + 1
        store_mask = store_offsets < n
        tl.store(a_ptr + store_offsets, result, mask=store_mask)

def s112_triton(a, b):
    n = a.shape[0]
    
    # Create read-only copy to avoid WAR race condition
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
    
    s112_kernel[grid](
        a,
        a_copy,
        b,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )