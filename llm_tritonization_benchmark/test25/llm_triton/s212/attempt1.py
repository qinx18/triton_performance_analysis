import triton
import triton.language as tl

@triton.jit
def s212_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < n
    
    # Load current elements
    a_vals = tl.load(a_copy_ptr + current_offsets, mask=mask)
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask)
    
    # Load a[i+1] elements for the second computation
    next_offsets = current_offsets + 1
    next_mask = next_offsets < (n + 1)  # We need up to n elements from original array
    a_next_vals = tl.load(a_copy_ptr + next_offsets, mask=next_mask)
    
    # Perform computations
    # a[i] *= c[i]
    new_a = a_vals * c_vals
    
    # b[i] += a[i + 1] * d[i]
    new_b = b_vals + a_next_vals * d_vals
    
    # Store results
    tl.store(a_ptr + current_offsets, new_a, mask=mask)
    tl.store(b_ptr + current_offsets, new_b, mask=mask)

def s212_triton(a, b, c, d):
    n = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s212_kernel[grid](
        a, a_copy, b, c, d,
        n,
        BLOCK_SIZE=BLOCK_SIZE
    )