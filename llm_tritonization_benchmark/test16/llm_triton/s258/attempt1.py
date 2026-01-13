import torch
import triton
import triton.language as tl

@triton.jit
def s258_expand_s_kernel(a_ptr, d_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Single block processes all elements sequentially to handle dependencies
    block_id = tl.program_id(0)
    if block_id != 0:
        return
    
    s_val = 0.0
    
    # Process in chunks for memory efficiency
    for start_idx in range(0, n_elements, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE)
        current_offsets = start_idx + offsets
        mask = current_offsets < n_elements
        
        # Load current chunk
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the chunk
        for i in range(BLOCK_SIZE):
            if start_idx + i >= n_elements:
                break
                
            offset = start_idx + i
            if a_vals[i] > 0.0:
                s_val = d_vals[i] * d_vals[i]
            
            tl.store(s_expanded_ptr + offset, s_val)

@triton.jit 
def s258_kernel(a_ptr, aa_ptr, b_ptr, c_ptr, d_ptr, e_ptr, s_expanded_ptr, 
                n_elements, aa_stride, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load values
    s_vals = tl.load(s_expanded_ptr + current_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
    aa_vals = tl.load(aa_ptr + current_offsets, mask=mask, other=0.0)
    
    # Compute results
    b_vals = s_vals * c_vals + d_vals
    e_vals = (s_vals + 1.0) * aa_vals
    
    # Store results
    tl.store(b_ptr + current_offsets, b_vals, mask=mask)
    tl.store(e_ptr + current_offsets, e_vals, mask=mask)

def s258_triton(a, aa, b, c, d, e):
    n_elements = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded scalar array
    s_expanded = torch.zeros(n_elements, dtype=torch.float32, device=a.device)
    
    # Phase 1: Expand scalar s
    grid = (1,)  # Single block to handle dependencies
    s258_expand_s_kernel[grid](
        a, d, s_expanded, n_elements, BLOCK_SIZE
    )
    
    # Phase 2: Parallel computation using expanded array
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s258_kernel[grid](
        a, aa, b, c, d, e, s_expanded,
        n_elements, aa.stride(0), BLOCK_SIZE
    )
    
    return b, e