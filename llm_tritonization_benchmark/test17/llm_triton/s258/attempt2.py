import torch
import triton
import triton.language as tl

@triton.jit
def s258_expand_s_kernel(a_ptr, d_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Single thread processes all elements sequentially to handle scalar expansion
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    s_val = 0.0
    
    # Process in blocks to avoid memory issues
    for block_start in range(0, n_elements, BLOCK_SIZE):
        block_end = tl.minimum(block_start + BLOCK_SIZE, n_elements)
        actual_size = block_end - block_start
        
        offsets = tl.arange(0, BLOCK_SIZE)
        current_offsets = block_start + offsets
        mask = offsets < actual_size
        
        # Load data for this block
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block
        for idx in range(BLOCK_SIZE):
            if idx < actual_size:
                a_val = tl.load(a_ptr + block_start + idx)
                d_val = tl.load(d_ptr + block_start + idx)
                
                if a_val > 0.0:
                    s_val = d_val * d_val
                
                tl.store(s_expanded_ptr + block_start + idx, s_val)

@triton.jit
def s258_compute_kernel(b_ptr, e_ptr, c_ptr, d_ptr, aa_ptr, s_expanded_ptr, 
                       n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load expanded s values
    s_vals = tl.load(s_expanded_ptr + current_offsets, mask=mask, other=0.0)
    
    # Load other arrays
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
    N = aa.shape[0]
    
    # Create expanded scalar array
    s_expanded = torch.zeros(N, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    
    # Phase 1: Expand scalar s with conditional propagation
    grid = (1,)  # Single thread handles sequential dependency
    s258_expand_s_kernel[grid](
        a, d, s_expanded, N, BLOCK_SIZE
    )
    
    # Phase 2: Compute arrays in parallel using expanded s
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s258_compute_kernel[grid](
        b, e, c, d, aa, s_expanded, N, BLOCK_SIZE
    )