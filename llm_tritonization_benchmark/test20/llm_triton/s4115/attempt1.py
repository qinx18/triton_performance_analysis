import triton
import triton.language as tl
import torch

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    local_sum = 0.0
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load a[i] values
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Load indices ip[i]
        ip_vals = tl.load(ip_ptr + current_offsets, mask=mask, other=0)
        
        # Load b[ip[i]] values using gathered indices
        b_vals = tl.load(b_ptr + ip_vals, mask=mask, other=0.0)
        
        # Compute products and accumulate
        products = a_vals * b_vals
        local_sum += tl.sum(products)
    
    # Store the final sum
    if tl.program_id(0) == 0:
        tl.store(output_ptr, local_sum)

def s4115_triton(a, b, ip):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create output tensor for the sum
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single program
    grid = (1,)
    s4115_kernel[grid](
        a, b, ip, output,
        n, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()