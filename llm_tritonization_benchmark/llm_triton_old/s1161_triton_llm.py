import torch
import triton
import triton.language as tl

@triton.jit
def s1161_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s1161 computation with conditional branching
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements (i < LEN_1D-1, so n_elements = LEN_1D-1)
    mask = offsets < n_elements
    
    # Load input arrays with masking
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + offsets, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Condition: c[i] >= 0
    condition = c_vals >= 0.0
    
    # When c[i] >= 0: a[i] = c[i] + d[i] * e[i]
    new_a = tl.where(condition, c_vals + d_vals * e_vals, a_vals)
    
    # When c[i] < 0: b[i] = a[i] + d[i] * d[i]
    # Note: use original a_vals for b computation, not new_a
    new_b = tl.where(~condition, a_vals + d_vals * d_vals, b_vals)
    
    # Store results
    tl.store(a_ptr + offsets, new_a, mask=mask)
    tl.store(b_ptr + offsets, new_b, mask=mask)

def s1161_triton(a, b, c, d, e):
    """
    Triton implementation of TSVC s1161 with optimized memory access patterns
    """
    # Ensure tensors are contiguous and on GPU
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    # Process elements 0 to LEN_1D-2 (i < LEN_1D-1)
    n_elements = c.shape[0] - 1
    
    if n_elements <= 0:
        return a, b
    
    # Choose block size for optimal memory coalescing
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 1024))
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s1161_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b