import torch
import triton
import triton.language as tl

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    partial_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load a[i]
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Load ip[i] 
        ip_vals = tl.load(ip_ptr + current_offsets, mask=mask, other=0)
        
        # Load b[ip[i]] - gather operation
        b_vals = tl.load(b_ptr + ip_vals, mask=mask, other=0.0)
        
        # Compute a[i] * b[ip[i]]
        products = a_vals * b_vals
        
        # Add to partial sum only for valid elements
        partial_sum += tl.where(mask, products, 0.0)
    
    # Reduce partial sum to scalar
    total_sum = tl.sum(partial_sum)
    
    # Store result (only one thread stores)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, total_sum)

def s4115_triton(a, b, ip):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block since we need global reduction
    grid = (1,)
    
    s4115_kernel[grid](
        a, b, ip, result,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()