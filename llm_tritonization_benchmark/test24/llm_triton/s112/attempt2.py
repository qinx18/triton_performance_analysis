import triton
import triton.language as tl

@triton.jit
def s112_kernel(a_ptr, a_copy_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load from read-only copy and b
    a_vals = tl.load(a_copy_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute a[i] + b[i]
    result = a_vals + b_vals
    
    # Store to a[i+1] (shift by +1)
    store_indices = indices + 1
    store_mask = mask
    
    tl.store(a_ptr + store_indices, result, mask=store_mask)

def s112_triton(a, b):
    n = a.shape[0]
    n_elements = n - 1  # Process indices 0 to n-2
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s112_kernel[grid](
        a, a_copy, b, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )