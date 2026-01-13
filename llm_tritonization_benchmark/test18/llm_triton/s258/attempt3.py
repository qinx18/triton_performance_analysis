import torch
import triton
import triton.language as tl

@triton.jit
def s258_expand_s_kernel(a_ptr, d_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    # Only use the first thread block to process sequentially
    if pid != 0:
        return
    
    s_val = 0.0
    
    # Process all elements sequentially to maintain dependency
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_idx in range(num_blocks):
        block_start = block_idx * BLOCK_SIZE
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values for this block
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block sequentially
        for i in range(BLOCK_SIZE):
            idx = block_start + i
            if idx < n_elements:
                # Check condition and update s if needed
                if a_vals[i] > 0.0:
                    s_val = d_vals[i] * d_vals[i]
                
                # Store expanded s value
                tl.store(s_expanded_ptr + idx, s_val)

@triton.jit
def s258_compute_kernel(s_expanded_ptr, c_ptr, d_ptr, aa_ptr, b_ptr, e_ptr, 
                       n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
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
    n_elements = aa.shape[0]  # Use aa.shape[0] for LEN_2D
    
    # Create expanded scalar array
    s_expanded = torch.zeros(n_elements, dtype=torch.float32, device=a.device)
    
    BLOCK_SIZE = 256
    
    # Phase 1: Expand scalar s
    grid_expand = (1,)  # Only one thread block needed
    s258_expand_s_kernel[grid_expand](
        a, d, s_expanded,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 2: Compute final results in parallel
    grid_compute = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s258_compute_kernel[grid_compute](
        s_expanded, c, d, aa[0], b, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )