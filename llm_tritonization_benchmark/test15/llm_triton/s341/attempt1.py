import torch
import triton
import triton.language as tl

@triton.jit
def s341_kernel(b_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load b values
    b_offsets = block_start + offsets
    mask = b_offsets < n_elements
    b_vals = tl.load(b_ptr + b_offsets, mask=mask, other=0.0)
    
    # Compute condition mask
    condition_mask = b_vals > 0.0
    
    # Store condition results back to global memory for later processing
    # We'll use a separate array to store the conditions
    condition_ptr = a_ptr + n_elements  # Use space after a array
    tl.store(condition_ptr + b_offsets, condition_mask.to(tl.float32), mask=mask)
    
    # Store b values that meet condition
    tl.store(condition_ptr + n_elements + b_offsets, b_vals, mask=mask)

def s341_triton(a, b):
    n_elements = b.numel()
    BLOCK_SIZE = 256
    
    # Allocate extra space for temporary storage
    temp_size = n_elements * 2
    temp_buffer = torch.zeros(temp_size, dtype=a.dtype, device=a.device)
    extended_a = torch.cat([a, temp_buffer])
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel to compute conditions
    s341_kernel[grid](
        b, extended_a,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Extract condition mask and values
    condition_start = n_elements
    values_start = n_elements * 2
    
    conditions = extended_a[condition_start:condition_start + n_elements]
    stored_values = extended_a[values_start:values_start + n_elements]
    
    # Convert to boolean mask
    mask = conditions > 0.5
    
    # Pack positive values using PyTorch's boolean indexing
    packed_values = b[mask]
    num_packed = packed_values.numel()
    
    if num_packed > 0:
        a[:num_packed] = packed_values