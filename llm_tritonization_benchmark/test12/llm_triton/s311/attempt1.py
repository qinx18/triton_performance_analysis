import torch
import triton
import triton.language as tl

@triton.jit
def s311_kernel(a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    sum_val = tl.zeros((1,), dtype=tl.float32)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        sum_val += tl.sum(a_vals)
    
    tl.store(tl.program_id(0) * 1 + tl.arange(0, 1), sum_val)

def s311_triton(a):
    n_elements = a.numel()
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_elements))
    
    # Create output tensor for the sum
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single program
    s311_kernel[(1,)](
        a, n_elements, BLOCK_SIZE
    )
    
    # Perform final reduction on GPU
    sum_result = torch.sum(a)
    
    return sum_result