import torch
import triton
import triton.language as tl

@triton.jit
def s277_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Load b[i+1] for writing
    b_next_mask = (idx + 1) < (n_elements + 1)
    
    # Control flow logic
    # if (a[i] >= 0.) goto L20;
    a_nonneg = a_vals >= 0.0
    
    # if (b[i] >= 0.) goto L30;  
    b_nonneg = b_vals >= 0.0
    
    # a[i] += c[i] * d[i]; (only if both a[i] < 0 and b[i] < 0)
    should_update_a = (~a_nonneg) & (~b_nonneg)
    new_a = tl.where(should_update_a, a_vals + c_vals * d_vals, a_vals)
    
    # L30: b[i+1] = c[i] + d[i] * e[i]; (only if a[i] < 0)
    should_update_b = ~a_nonneg
    new_b_next = c_vals + d_vals * e_vals
    
    # Store results
    tl.store(a_ptr + idx, new_a, mask=mask)
    tl.store(b_ptr + idx + 1, new_b_next, mask=mask & b_next_mask & should_update_b)

def s277_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Process in strips of size 1 due to RAW dependency
    for strip_start in range(0, n_elements, 1):
        strip_end = min(strip_start + 1, n_elements)
        if strip_start >= n_elements:
            break
            
        # Single element processing
        idx = strip_start
        
        # Control flow logic
        if a[idx] >= 0.0:
            continue  # goto L20 - do nothing
            
        if b[idx] >= 0.0:
            # goto L30 - only update b[i+1]
            b[idx + 1] = c[idx] + d[idx] * e[idx]
            continue
            
        # Both conditions false - update a[i] then fall through to L30
        a[idx] += c[idx] * d[idx]
        # L30: update b[i+1]  
        b[idx + 1] = c[idx] + d[idx] * e[idx]