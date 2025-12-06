import torch
import triton
import triton.language as tl

@triton.jit
def s161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Condition: b[i] < 0
    condition = b_vals < 0.0
    
    # Branch 1: a[i] = c[i] + d[i] * e[i]
    branch1_result = c_vals + d_vals * e_vals
    
    # Branch 2: c[i+1] = a[i] + d[i] * d[i]
    branch2_result = a_vals + d_vals * d_vals
    
    # Store a[i] for branch 1 (when b[i] >= 0)
    a_store_mask = mask & (condition == False)
    tl.store(a_ptr + idx, branch1_result, mask=a_store_mask)
    
    # Store c[i+1] for branch 2 (when b[i] < 0)
    c_store_idx = idx + 1
    c_store_mask = mask & condition & (c_store_idx < (n_elements + 1))
    tl.store(c_ptr + c_store_idx, branch2_result, mask=c_store_mask)

def s161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 1
    
    # Process sequentially due to loop-carried dependency
    for i in range(0, n_elements, BLOCK_SIZE):
        remaining = min(BLOCK_SIZE, n_elements - i)
        
        # Create views for current strip
        a_view = a[i:i+remaining]
        b_view = b[i:i+remaining]
        c_view = c[i:]  # Need extra element for c[i+1]
        d_view = d[i:i+remaining]
        e_view = e[i:i+remaining]
        
        grid = (1,)
        s161_kernel[grid](
            a_view, b_view, c_view, d_view, e_view,
            remaining, BLOCK_SIZE
        )