import torch
import triton
import triton.language as tl

@triton.jit
def s251_expand_s_kernel(s_expanded_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Single program processes all elements sequentially
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    # Process all elements sequentially to handle scalar expansion
    for block_start in range(0, n_elements, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE)
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute s values for this block
        s_vals = b_vals + c_vals * d_vals
        
        # Store s values
        tl.store(s_expanded_ptr + current_offsets, s_vals, mask=mask)

@triton.jit
def s251_compute_kernel(a_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load expanded s values
    s_vals = tl.load(s_expanded_ptr + current_offsets, mask=mask, other=0.0)
    
    # Compute a[i] = s * s
    a_vals = s_vals * s_vals
    
    # Store result
    tl.store(a_ptr + current_offsets, a_vals, mask=mask)

def s251_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded scalar array
    s_expanded = torch.zeros_like(a)
    
    # Phase 1: Expand scalar s
    grid_expand = (1,)  # Single program for sequential processing
    s251_expand_s_kernel[grid_expand](
        s_expanded,
        b,
        c,
        d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 2: Compute a[i] = s * s in parallel
    grid_compute = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s251_compute_kernel[grid_compute](
        a,
        s_expanded,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a