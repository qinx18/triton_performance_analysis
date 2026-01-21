import triton
import triton.language as tl
import torch

@triton.jit
def s2251_expand_kernel(b_ptr, c_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    if pid == 0:
        # Single thread processes all elements sequentially
        s_val = 0.0
        for i in range(n_elements):
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            # First store the current s value (for use in a[i] = s*e[i])
            tl.store(s_expanded_ptr + i, s_val)
            # Then update s for next iteration
            s_val = b_val + c_val

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load arrays
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    s_vals = tl.load(s_expanded_ptr + offsets, mask=mask)
    
    # Compute a[i] = s*e[i]
    a_vals = s_vals * e_vals
    
    # Compute b[i] = a[i]+d[i]
    new_b_vals = a_vals + d_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    tl.store(b_ptr + offsets, new_b_vals, mask=mask)

def s2251_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded scalar array
    s_expanded = torch.zeros_like(a)
    
    # Phase 1: Expand scalar s
    grid = (1,)  # Single block for sequential processing
    s2251_expand_kernel[grid](b, c, s_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Parallel computation
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s2251_kernel[grid](a, b, c, d, e, s_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE)