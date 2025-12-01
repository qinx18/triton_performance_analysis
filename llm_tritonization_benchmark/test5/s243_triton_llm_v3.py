import triton
import triton.language as tl

@triton.jit
def s243_kernel(
    a_ptr,
    a_copy_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load from original arrays
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask)
        
        # Load a[i+1] from copy for third statement
        next_offsets = current_offsets + 1
        next_mask = next_offsets < n_elements
        a_next_vals = tl.load(a_copy_ptr + next_offsets, mask=next_mask, other=0.0)
        
        # First statement: a[i] = b[i] + c[i] * d[i]
        a_vals_1 = b_vals + c_vals * d_vals
        tl.store(a_ptr + current_offsets, a_vals_1, mask=mask)
        
        # Second statement: b[i] = a[i] + d[i] * e[i]
        b_vals_new = a_vals_1 + d_vals * e_vals
        tl.store(b_ptr + current_offsets, b_vals_new, mask=mask)
        
        # Third statement: a[i] = b[i] + a[i+1] * d[i]
        a_vals_2 = b_vals_new + a_next_vals * d_vals
        tl.store(a_ptr + current_offsets, a_vals_2, mask=mask)

def s243_triton(a, b, c, d, e):
    n_elements = len(a) - 1  # Loop goes to LEN_1D-1
    BLOCK_SIZE = 256
    
    # Create read-only copy for WAR dependency
    a_copy = a.clone()
    
    grid = (1,)
    s243_kernel[grid](
        a, a_copy, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )