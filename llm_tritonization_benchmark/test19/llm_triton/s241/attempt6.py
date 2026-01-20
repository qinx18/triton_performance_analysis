import triton
import triton.language as tl

@triton.jit
def s241_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    mask_plus_1 = (offsets + 1) < (n_elements + 1)
    
    # Load values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    a_plus1_vals = tl.load(a_copy_ptr + offsets + 1, mask=mask_plus_1)
    
    # First computation: a[i] = b[i] * c[i] * d[i]
    a_vals = b_vals * c_vals * d_vals
    
    # Store a values
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    
    # Second computation: b[i] = a[i] * a[i+1] * d[i]
    b_new_vals = a_vals * a_plus1_vals * d_vals
    
    # Store b values
    tl.store(b_ptr + offsets, b_new_vals, mask=mask)

def s241_triton(a, b, c, d):
    n_elements = a.shape[0] - 1
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s241_kernel[grid](
        a, a_copy, b, c, d, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )