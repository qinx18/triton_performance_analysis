import torch
import triton
import triton.language as tl

@triton.jit
def s253_expand_s_kernel(a_ptr, b_ptr, d_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load previous s values (from previous blocks)
    s_vals = tl.load(s_expanded_ptr + offsets, mask=mask, other=0.0)
    
    # Get the last valid s value from previous block as starting point
    prev_s = 0.0
    if pid > 0:
        prev_s = tl.load(s_expanded_ptr + block_start - 1)
    
    # Load input arrays
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    
    # Compute conditions and new s values
    condition = a_vals > b_vals
    new_s_vals = a_vals - b_vals * d_vals
    
    # Sequential scan within block
    result_s = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    current_s = prev_s
    
    # Process each element sequentially within the block
    for i in tl.static_range(BLOCK_SIZE):
        element_idx = block_start + i
        if element_idx < n_elements:
            # Extract scalar values
            cond_val = tl.load(a_ptr + element_idx) > tl.load(b_ptr + element_idx)
            if cond_val:
                current_s = tl.load(a_ptr + element_idx) - tl.load(b_ptr + element_idx) * tl.load(d_ptr + element_idx)
        
        # Store current s value at position i
        tl.store(s_expanded_ptr + block_start + i, current_s, mask=(block_start + i) < n_elements)

@triton.jit
def s253_kernel(a_ptr, b_ptr, c_ptr, d_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    s_vals = tl.load(s_expanded_ptr + offsets, mask=mask, other=0.0)
    
    # Apply conditional updates
    condition = a_vals > b_vals
    
    # Update c where condition is true
    c_updated = tl.where(condition, c_vals + s_vals, c_vals)
    
    # Update a where condition is true
    a_updated = tl.where(condition, s_vals, a_vals)
    
    # Store results
    tl.store(c_ptr + offsets, c_updated, mask=mask)
    tl.store(a_ptr + offsets, a_updated, mask=mask)

def s253_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded scalar array
    s_expanded = torch.zeros_like(a)
    
    # Phase 1: Expand scalar s with multiple blocks handling scan
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    for _ in range(2):  # Multiple passes to propagate values across blocks
        s253_expand_s_kernel[grid](a, b, d, s_expanded, n_elements, BLOCK_SIZE)
    
    # Phase 2: Apply updates in parallel
    s253_kernel[grid](a, b, c, d, s_expanded, n_elements, BLOCK_SIZE)