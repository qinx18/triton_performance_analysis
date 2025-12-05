import torch
import triton
import triton.language as tl

@triton.jit
def s277_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, strip_start, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + block_start + offsets
    
    mask = (block_start + offsets) < BLOCK_SIZE
    
    # Load values
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Control flow logic
    a_nonneg = a_vals >= 0.0
    b_nonneg = b_vals >= 0.0
    
    # Update a[i] only if both a[i] < 0 and b[i] < 0
    should_update_a = (~a_nonneg) & (~b_nonneg)
    new_a = tl.where(should_update_a, a_vals + c_vals * d_vals, a_vals)
    
    # Update b[i+1] if a[i] < 0 (either branch that reaches L30)
    should_update_b = ~a_nonneg
    new_b_next = c_vals + d_vals * e_vals
    
    # Store results
    tl.store(a_ptr + idx, new_a, mask=mask)
    
    # Store to b[i+1] with proper bounds checking
    next_idx = idx + 1
    next_mask = mask & should_update_b & (next_idx < a_ptr.numel())
    tl.store(b_ptr + next_idx, new_b_next, mask=next_mask)

def s277_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1
    
    # Process each element sequentially due to RAW dependency
    for i in range(n_elements):
        s277_kernel[(1,)](
            a, b, c, d, e,
            strip_start=i,
            BLOCK_SIZE=1
        )