import triton
import triton.language as tl
import torch

@triton.jit
def expand_s_kernel(b_ptr, c_ptr, s_expanded_ptr, n_elements):
    # Single thread processes all elements sequentially
    s_val = 0.0  # Initial value
    for i in range(n_elements):
        # Load b[i] and c[i]
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        # Update s value
        s_val = b_val + c_val
        # Store expanded value
        tl.store(s_expanded_ptr + i, s_val)

@triton.jit
def s1251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load expanded s values
    s_vals = tl.load(s_expanded_ptr + offsets, mask=mask)
    
    # Load arrays
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Compute: b[i] = a[i] + d[i]
    b_vals = a_vals + d_vals
    tl.store(b_ptr + offsets, b_vals, mask=mask)
    
    # Compute: a[i] = s * e[i]
    a_vals = s_vals * e_vals
    tl.store(a_ptr + offsets, a_vals, mask=mask)

def s1251_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded scalar array
    s_expanded = torch.zeros(n_elements, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar s
    expand_s_kernel[(1,)](b, c, s_expanded, n_elements)
    
    # Phase 2: Parallel computation using expanded array
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s1251_kernel[grid](a, b, c, d, e, s_expanded, n_elements, BLOCK_SIZE)