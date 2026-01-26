import triton
import triton.language as tl
import torch

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    sum_val = 0.0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load a values
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Load indices
        indices = tl.load(ip_ptr + current_offsets, mask=mask, other=0)
        
        # Load b values using indirect addressing
        b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
        
        # Compute products and accumulate
        products = a_vals * b_vals
        sum_val += tl.sum(products)
    
    tl.store(output_ptr, sum_val)

def s4115_triton(a, b, ip):
    n_elements = a.shape[0]
    
    # Create output tensor for the sum
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    
    s4115_kernel[(1,)](
        a, b, ip, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()