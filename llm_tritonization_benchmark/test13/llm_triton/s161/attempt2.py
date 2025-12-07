import torch
import triton
import triton.language as tl

@triton.jit
def s161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    strip_id = tl.program_id(0)
    
    if strip_id >= n_elements:
        return
    
    # Each strip processes exactly 1 element
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_id + offsets
    
    # Load values
    mask = idx < n_elements
    b_val = tl.load(b_ptr + idx, mask=mask)
    a_val = tl.load(a_ptr + idx, mask=mask)
    c_val = tl.load(c_ptr + idx, mask=mask)
    d_val = tl.load(d_ptr + idx, mask=mask)
    e_val = tl.load(e_ptr + idx, mask=mask)
    
    # Conditional logic: if (b[i] < 0.0)
    condition = b_val < 0.0
    
    # Path 1 (condition False): a[i] = c[i] + d[i] * e[i]
    result_a = c_val + d_val * e_val
    
    # Path 2 (condition True): c[i+1] = a[i] + d[i] * d[i]
    result_c = a_val + d_val * d_val
    
    # Store results based on condition
    # When condition is False: update a[i]
    tl.store(a_ptr + idx, result_a, mask=mask & (~condition))
    
    # When condition is True: update c[i+1]
    next_idx = idx + 1
    next_mask = condition & mask
    tl.store(c_ptr + next_idx, result_c, mask=next_mask)

def s161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    # Must process strips sequentially due to RAW dependency
    BLOCK_SIZE = 1
    
    for strip_start in range(0, n_elements):
        grid = (1,)
        s161_kernel[grid](
            a, b, c, d, e,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )