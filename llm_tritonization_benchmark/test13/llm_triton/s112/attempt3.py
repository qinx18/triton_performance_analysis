import torch
import triton
import triton.language as tl

@triton.jit
def s112_kernel(a_ptr, a_copy_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load from read-only copy and b array
        a_vals = tl.load(a_copy_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute a[i] + b[i]
        result = a_vals + b_vals
        
        # Store to a[i+1] (offset by 1)
        store_offsets = current_offsets + 1
        store_mask = mask
        tl.store(a_ptr + store_offsets, result, mask=store_mask)

def s112_triton(a, b):
    n = a.shape[0] - 1  # Process elements from LEN_1D-2 down to 0
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s112_kernel[grid](
        a,
        a_copy,
        b,
        n,
        BLOCK_SIZE=BLOCK_SIZE
    )