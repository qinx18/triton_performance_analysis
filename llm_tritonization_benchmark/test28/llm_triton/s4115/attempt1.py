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
        
        # Load a[i]
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Load indices ip[i]
        ip_vals = tl.load(ip_ptr + current_offsets, mask=mask, other=0)
        
        # Gather b[ip[i]] - need to handle each element
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                idx = tl.load(ip_ptr + block_start + i)
                b_val = tl.load(b_ptr + idx)
                a_val = tl.load(a_ptr + block_start + i)
                sum_val += a_val * b_val
    
    # Store result
    if tl.program_id(0) == 0:
        tl.store(output_ptr, sum_val)

@triton.jit
def s4115_kernel_simple(a_ptr, b_ptr, ip_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:
        sum_val = 0.0
        
        for i in range(n_elements):
            a_val = tl.load(a_ptr + i)
            idx = tl.load(ip_ptr + i)
            b_val = tl.load(b_ptr + idx)
            sum_val += a_val * b_val
            
        tl.store(output_ptr, sum_val)

def s4115_triton(a, b, ip):
    n_elements = a.shape[0]
    
    # Create output tensor
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    
    # Use simple kernel since this is a reduction operation
    grid = (1,)
    
    s4115_kernel_simple[grid](
        a, b, ip, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()