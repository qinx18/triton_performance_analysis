import triton
import triton.language as tl
import torch

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    partial_sum = 0.0
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load a[i]
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Load indices ip[i]
        indices = tl.load(ip_ptr + current_offsets, mask=mask, other=0)
        
        # Load b[ip[i]] - gather operation
        b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
        
        # Compute a[i] * b[ip[i]]
        products = a_vals * b_vals
        
        # Sum the products
        partial_sum += tl.sum(products)
    
    # Store the partial sum (only first thread writes)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, partial_sum)

def s4115_triton(a, b, ip):
    n = a.shape[0]
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s4115_kernel[grid](
        a, b, ip, result,
        n, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()