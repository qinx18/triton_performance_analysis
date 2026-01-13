import torch
import triton
import triton.language as tl

@triton.jit
def s319_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    block_sum = 0.0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load c, d, e values
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute a[i] = c[i] + d[i] and accumulate sum
        a_vals = c_vals + d_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        block_sum += tl.sum(tl.where(mask, a_vals, 0.0))
        
        # Compute b[i] = c[i] + e[i] and accumulate sum
        b_vals = c_vals + e_vals
        tl.store(b_ptr + current_offsets, b_vals, mask=mask)
        block_sum += tl.sum(tl.where(mask, b_vals, 0.0))
    
    # Store the block sum in a temporary tensor for reduction outside kernel
    tl.store(a_ptr, block_sum)

def s319_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    
    # Create a temporary tensor for the sum result
    temp_sum = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch single block to handle entire array and reduction
    s319_kernel[(1,)](
        temp_sum, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return temp_sum.item()