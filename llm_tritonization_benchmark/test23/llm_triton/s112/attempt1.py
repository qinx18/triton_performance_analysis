import triton
import triton.language as tl

@triton.jit
def s112_kernel(a, a_copy, b, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n
    
    # Load from read-only copy and b array
    a_vals = tl.load(a_copy + indices, mask=mask)
    b_vals = tl.load(b + indices, mask=mask)
    
    # Compute a[i+1] = a[i] + b[i] (storing at i+1)
    result = a_vals + b_vals
    store_indices = indices + 1
    store_mask = mask & (store_indices < n)
    
    # Store to original array at offset position
    tl.store(a + store_indices, result, mask=store_mask)

def s112_triton(a, b):
    n = a.shape[0] - 1  # Process n-1 elements since we access i+1
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s112_kernel[grid](a, a_copy, b, n, BLOCK_SIZE=BLOCK_SIZE)