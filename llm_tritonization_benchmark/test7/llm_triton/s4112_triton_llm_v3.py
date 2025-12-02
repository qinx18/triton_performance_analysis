import torch
import triton
import triton.language as tl

@triton.jit
def s4112_kernel(
    a_ptr,
    b_ptr, 
    ip_ptr,
    s,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    # Create mask for valid elements
    mask = i_offsets < n_elements
    
    # Load indices from ip array
    indices = tl.load(ip_ptr + i_offsets, mask=mask, other=0)
    
    # Load current values from a array
    a_vals = tl.load(a_ptr + i_offsets, mask=mask, other=0.0)
    
    # Load values from b array using indirect indexing
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Compute a[i] += b[ip[i]] * s
    result = a_vals + b_vals * s
    
    # Store result back to a array
    tl.store(a_ptr + i_offsets, result, mask=mask)

def s4112_triton(a, b, ip, s):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s4112_kernel[grid](
        a,
        b,
        ip,
        s,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a