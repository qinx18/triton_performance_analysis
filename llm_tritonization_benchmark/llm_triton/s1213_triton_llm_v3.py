import triton
import triton.language as tl
import torch

@triton.jit
def s1213_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must handle the dependency between the two statements
    # a[i] = b[i-1] + c[i]
    # b[i] = a[i+1] * d[i]
    # The second statement reads a[i+1] which will be written in the next iteration
    
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid indices (1 <= i < n_elements-1)
    mask = (offsets >= 1) & (offsets < n_elements - 1)
    
    # Load arrays with bounds checking
    # For a[i+1], we need to load one element ahead
    b_prev = tl.load(b_ptr + offsets - 1, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # For a[i+1] in the second statement, we need to read the value that will be computed
    # Since we're computing sequentially and a[i+1] depends on future computation,
    # we need to handle this dependency properly
    a_next_mask = (offsets + 1 < n_elements - 1) & (offsets >= 1)
    
    # First statement: a[i] = b[i-1] + c[i]
    a_new = b_prev + c_vals
    
    # Store a[i] first
    tl.store(a_ptr + offsets, a_new, mask=mask)
    
    # For the second statement, we need a[i+1]
    # Load the current a[i+1] values (these will be the old values before update)
    a_next = tl.load(a_ptr + offsets + 1, mask=a_next_mask)
    
    # However, due to the dependency, we need to use the updated values
    # This is tricky in parallel - we need to compute what a[i+1] will be
    b_curr = tl.load(b_ptr + offsets, mask=a_next_mask)  # b[i] for a[i+1] computation
    c_next = tl.load(c_ptr + offsets + 1, mask=a_next_mask)  # c[i+1] for a[i+1] computation
    a_next_updated = b_curr + c_next
    
    # Use the mask to select between updated and boundary values
    a_next_final = tl.where(a_next_mask, a_next_updated, a_next)
    
    # Second statement: b[i] = a[i+1] * d[i]
    b_new = a_next_final * d_vals
    
    # Store b[i]
    tl.store(b_ptr + offsets, b_new, mask=mask)

def s1213_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s1213_kernel[grid](
        a, b, c, d, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )