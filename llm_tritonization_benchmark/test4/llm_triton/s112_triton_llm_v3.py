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
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load from copy for a[i] and b[i]
        a_vals = tl.load(a_copy_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        # Compute a[i] + b[i]
        result = a_vals + b_vals
        
        # Store to a[i+1] (original array)
        write_offsets = current_offsets + 1
        write_mask = write_offsets < (n + 1)
        tl.store(a_ptr + write_offsets, result, mask=write_mask)

def s112_triton(a, b):
    n = a.shape[0] - 1  # Loop from LEN_1D - 2 down to 0
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s112_kernel[grid](
        a,
        a_copy,
        b,
        n,
        BLOCK_SIZE,
    )