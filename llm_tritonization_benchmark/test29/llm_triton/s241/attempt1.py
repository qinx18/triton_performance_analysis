import triton
import triton.language as tl
import torch

@triton.jit
def s241_kernel(
    a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate number of blocks needed
    n_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    
    # Process each block sequentially to maintain dependencies
    for block_idx in range(n_blocks):
        block_start = block_idx * BLOCK_SIZE
        offsets = tl.arange(0, BLOCK_SIZE)
        current_offsets = block_start + offsets
        
        # Mask for valid elements
        mask = current_offsets < n_elements
        
        # Load values for first statement: a[i] = b[i] * c[i] * d[i]
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Compute a[i] = b[i] * c[i] * d[i]
        a_new = b_vals * c_vals * d_vals
        
        # Store new a values
        tl.store(a_ptr + current_offsets, a_new, mask=mask)
        
        # Load values for second statement: b[i] = a[i] * a[i+1] * d[i]
        # Use read-only copy for a[i] and compute a[i+1] offsets
        a_i_vals = tl.load(a_copy_ptr + current_offsets, mask=mask)
        
        # For a[i+1], we need to load from offset+1, but mask appropriately
        next_offsets = current_offsets + 1
        next_mask = mask & (next_offsets < n_elements)
        a_i_plus_1_vals = tl.load(a_copy_ptr + next_offsets, mask=next_mask)
        
        # Compute b[i] = a[i] * a[i+1] * d[i]
        # Only compute where we have valid a[i+1] (i.e., next_mask)
        b_new = a_i_vals * a_i_plus_1_vals * d_vals
        
        # Store new b values (use next_mask since we need valid a[i+1])
        tl.store(b_ptr + current_offsets, b_new, mask=next_mask)

def s241_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Loop bound is LEN_1D-1
    
    # Create read-only copy of array a to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    
    # Launch kernel with sequential block processing
    grid = (1,)  # Single thread block to maintain sequential execution
    
    s241_kernel[grid](
        a, a_copy, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )