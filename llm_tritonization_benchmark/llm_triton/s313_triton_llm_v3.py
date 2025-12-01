import torch
import triton
import triton.language as tl

@triton.jit
def s313_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    dot_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        products = a_vals * b_vals
        dot_sum += tl.where(mask, products, 0.0)
    
    total_dot = tl.sum(dot_sum, axis=0)
    
    if tl.program_id(0) == 0:
        tl.store(output_ptr, total_dot)

def s313_triton(a, b):
    n_elements = a.shape[0]
    
    output = torch.zeros(1, dtype=torch.float32, device=a.device)
    
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s313_kernel[grid](
        a, b, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()