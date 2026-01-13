import torch
import triton
import triton.language as tl

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a reduction kernel, so we use one program to compute the entire sum
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    # Process elements in blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    partial_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load a[i]
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Load ip[i] indices
        ip_vals = tl.load(ip_ptr + current_offsets, mask=mask, other=0)
        
        # Load b[ip[i]] using gathered indices
        b_vals = tl.load(b_ptr + ip_vals, mask=mask, other=0.0)
        
        # Compute a[i] * b[ip[i]]
        products = a_vals * b_vals
        partial_sum += tl.where(mask, products, 0.0)
    
    # Reduce the partial sum to a scalar
    total_sum = tl.sum(partial_sum)
    
    # Store the result
    tl.store(output_ptr, total_sum)

def s4115_triton(a, b, ip):
    n_elements = a.shape[0]
    
    # Output tensor to store the sum
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch parameters
    BLOCK_SIZE = 1024
    grid = (1,)  # Single program to compute the reduction
    
    s4115_kernel[grid](
        a, b, ip, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()