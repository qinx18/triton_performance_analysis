import triton
import triton.language as tl
import torch

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    partial_sum = 0.0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load a[i]
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Load ip[i] indices
        ip_vals = tl.load(ip_ptr + current_offsets, mask=mask, other=0)
        
        # Load b[ip[i]] using gather
        b_vals = tl.load(b_ptr + ip_vals, mask=mask, other=0.0)
        
        # Compute partial products and sum
        products = a_vals * b_vals
        partial_sum += tl.sum(products, axis=0)
    
    tl.store(output_ptr, partial_sum)

def s4115_triton(a, b, ip):
    n_elements = a.shape[0]
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_elements))
    
    # Output tensor to store the final sum
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block to compute global sum
    grid = (1,)
    s4115_kernel[grid](
        a, b, ip, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()