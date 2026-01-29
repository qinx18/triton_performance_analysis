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
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        ip_vals = tl.load(ip_ptr + current_offsets, mask=mask, other=0)
        
        b_vals = tl.load(b_ptr + ip_vals, mask=mask, other=0.0)
        
        products = a_vals * b_vals
        sum_val += tl.sum(products)
    
    tl.store(output_ptr, sum_val)

def s4115_triton(a, b, ip):
    N = a.shape[0]
    
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    
    s4115_kernel[(1,)](
        a, b, ip, output,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()