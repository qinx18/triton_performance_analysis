import triton
import triton.language as tl

@triton.jit
def s212_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # First operation: a[i] *= c[i]
    a_new = a_vals * c_vals
    
    # Store updated a values
    tl.store(a_ptr + indices, a_new, mask=mask)
    
    # For second operation: b[i] += a[i + 1] * d[i]
    # Load a[i + 1] from the read-only copy
    indices_plus_1 = indices + 1
    mask_plus_1 = indices_plus_1 < (n_elements + 1)  # Allow reading a[n_elements-1+1] = a[n_elements]
    
    a_copy_plus_1 = tl.load(a_copy_ptr + indices_plus_1, mask=mask_plus_1, other=0.0)
    
    # Update b[i]
    b_new = b_vals + a_copy_plus_1 * d_vals
    
    # Store updated b values
    tl.store(b_ptr + indices, b_new, mask=mask)

def s212_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Loop goes from 0 to LEN_1D-2 (inclusive)
    
    # Create read-only copy of array a to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s212_kernel[grid](
        a, a_copy, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )