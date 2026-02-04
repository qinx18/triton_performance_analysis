import triton
import triton.language as tl
import torch

@triton.jit
def s221_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    
    # Create offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Create mask for valid indices and ensure i >= 1
    mask = (indices < n_elements) & (indices >= 1)
    
    # Load values
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + indices, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    
    # First statement: a[i] += c[i] * d[i]
    new_a_vals = a_vals + c_vals * d_vals
    tl.store(a_ptr + indices, new_a_vals, mask=mask)
    
    # Second statement: b[i] = b[i-1] + a[i] + d[i]
    # Load b[i-1] values
    prev_indices = indices - 1
    prev_mask = (indices < n_elements) & (indices >= 1) & (prev_indices >= 0)
    b_prev_vals = tl.load(b_ptr + prev_indices, mask=prev_mask, other=0.0)
    
    new_b_vals = b_prev_vals + new_a_vals + d_vals
    tl.store(b_ptr + indices, new_b_vals, mask=mask)

def s221_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # This loop has sequential dependency in b[i] = b[i-1] + ...
    # Must process sequentially
    BLOCK_SIZE = 128
    
    for start_idx in range(1, n_elements, BLOCK_SIZE):
        grid_size = 1
        remaining = min(BLOCK_SIZE, n_elements - start_idx)
        
        s221_kernel[(grid_size,)](
            a.data_ptr() - start_idx,  # Adjust pointer so indices work correctly
            b.data_ptr() - start_idx,
            c.data_ptr() - start_idx,
            d.data_ptr() - start_idx,
            start_idx + remaining,
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    return a, b