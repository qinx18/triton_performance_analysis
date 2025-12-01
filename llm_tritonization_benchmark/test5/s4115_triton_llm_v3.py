import torch
import triton
import triton.language as tl

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    sum_val = 0.0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        ip_vals = tl.load(ip_ptr + current_offsets, mask=mask, other=0)
        
        b_vals = tl.load(b_ptr + ip_vals, mask=mask, other=0.0)
        
        products = a_vals * b_vals
        block_sum = tl.sum(products)
        sum_val += block_sum
    
    if tl.program_id(0) == 0:
        tl.store(result_ptr, sum_val)

def s4115_triton(a, b, ip):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    grid = (1,)
    s4115_kernel[grid](
        a, b, ip, result,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()