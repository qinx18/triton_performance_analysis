import triton
import triton.language as tl

@triton.jit
def s241_kernel1(a, b, c, d, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    b_vals = tl.load(b + offsets, mask=mask)
    c_vals = tl.load(c + offsets, mask=mask)
    d_vals = tl.load(d + offsets, mask=mask)
    
    result = b_vals * c_vals * d_vals
    tl.store(a + offsets, result, mask=mask)

@triton.jit
def s241_kernel2(a, a_copy, b, d, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load current a values
    a_vals = tl.load(a_copy + offsets, mask=mask)
    
    # Load a[i+1] values with bounds checking
    next_offsets = offsets + 1
    next_mask = next_offsets < (n_elements + 1)  # Allow access to a[n_elements]
    combined_mask = mask & next_mask
    
    a_next_vals = tl.load(a_copy + next_offsets, mask=combined_mask)
    d_vals = tl.load(d + offsets, mask=mask)
    
    # Compute result only for valid indices
    result = a_vals * a_next_vals * d_vals
    tl.store(b + offsets, result, mask=combined_mask)

def s241_triton(a, b, c, d):
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # First kernel: a[i] = b[i] * c[i] * d[i]
    s241_kernel1[grid](a, b, c, d, n_elements, BLOCK_SIZE)
    
    # Create read-only copy for second kernel
    a_copy = a.clone()
    
    # Second kernel: b[i] = a[i] * a[i+1] * d[i]
    s241_kernel2[grid](a, a_copy, b, d, n_elements, BLOCK_SIZE)