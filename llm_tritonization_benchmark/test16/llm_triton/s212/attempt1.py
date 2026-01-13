import triton
import triton.language as tl

@triton.jit
def s212_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # a[i] *= c[i]
    a_new = a_vals * c_vals
    
    # b[i] += a[i + 1] * d[i]
    # Load a[i+1] from read-only copy
    idx_plus1 = idx + 1
    mask_plus1 = idx_plus1 < (n_elements + 1)
    a_plus1_vals = tl.load(a_copy_ptr + idx_plus1, mask=mask_plus1)
    b_new = b_vals + a_plus1_vals * d_vals
    
    # Store results
    tl.store(a_ptr + idx, a_new, mask=mask)
    tl.store(b_ptr + idx, b_new, mask=mask)

def s212_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s212_kernel[grid](
        a, a_copy, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )