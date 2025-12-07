import torch
import triton
import triton.language as tl

@triton.jit
def s4115_kernel(
    a_ptr, b_ptr, ip_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # This is a sparse dot product with gather operation
    # We need to reduce across all elements, so we'll use one block
    pid = tl.program_id(0)
    
    sum_val = 0.0
    
    # Process elements in chunks
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load a[i]
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Load ip[i] indices
        ip_vals = tl.load(ip_ptr + current_offsets, mask=mask, other=0)
        
        # Load b[ip[i]] - gather operation
        b_vals = tl.load(b_ptr + ip_vals, mask=mask, other=0.0)
        
        # Compute a[i] * b[ip[i]]
        products = a_vals * b_vals
        
        # Sum the products
        sum_val += tl.sum(products)
    
    # Store the result
    if pid == 0:
        tl.store(a_ptr + n_elements, sum_val)

def s4115_triton(a, b, ip):
    n_elements = a.shape[0]
    
    # We need to store the result somewhere - append space to array a
    a_extended = torch.cat([a, torch.zeros(1, dtype=a.dtype, device=a.device)])
    
    BLOCK_SIZE = 256
    grid = (1,)  # Single block for reduction
    
    s4115_kernel[grid](
        a_extended, b, ip,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a_extended[-1].item()