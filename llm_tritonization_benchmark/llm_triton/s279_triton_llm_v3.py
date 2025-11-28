import triton
import triton.language as tl
import torch

@triton.jit
def s279_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all arrays
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # First condition: if (a[i] > 0.)
    cond1 = a_vals > 0.0
    
    # Path when a[i] <= 0 (not goto L20)
    b_new = -b_vals + d_vals * d_vals
    cond2 = b_new <= a_vals
    
    # Update c for case where a[i] <= 0 and b[i] > a[i]
    c_update1 = c_vals + d_vals * e_vals
    
    # Path when a[i] > 0 (goto L20)
    c_update2 = -c_vals + e_vals * e_vals
    
    # Select appropriate c values based on conditions
    # If a[i] > 0, use c_update2
    # If a[i] <= 0 and b[i] > a[i], use c_update1
    # Otherwise keep original c_vals
    c_final = tl.where(cond1, c_update2, 
                      tl.where(~cond2, c_update1, c_vals))
    
    # Update b only when a[i] <= 0
    b_final = tl.where(cond1, b_vals, b_new)
    
    # Final update: a[i] = b[i] + c[i] * d[i]
    a_final = b_final + c_final * d_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_final, mask=mask)
    tl.store(b_ptr + offsets, b_final, mask=mask)
    tl.store(c_ptr + offsets, c_final, mask=mask)

def s279_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    s279_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )