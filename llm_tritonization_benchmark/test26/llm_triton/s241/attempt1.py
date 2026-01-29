import triton
import triton.language as tl

@triton.jit
def s241_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    a_copy_vals = tl.load(a_copy_ptr + indices, mask=mask)
    
    # Load a[i+1] values for the second equation
    indices_plus_1 = indices + 1
    mask_plus_1 = indices_plus_1 < (n_elements + 1)
    a_copy_vals_plus_1 = tl.load(a_copy_ptr + indices_plus_1, mask=mask_plus_1)
    
    # Compute a[i] = b[i] * c[i] * d[i]
    new_a_vals = b_vals * c_vals * d_vals
    
    # Compute b[i] = a[i] * a[i+1] * d[i]
    new_b_vals = new_a_vals * a_copy_vals_plus_1 * d_vals
    
    # Store results
    tl.store(a_ptr + indices, new_a_vals, mask=mask)
    tl.store(b_ptr + indices, new_b_vals, mask=mask)

def s241_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s241_kernel[grid](
        a, a_copy, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )