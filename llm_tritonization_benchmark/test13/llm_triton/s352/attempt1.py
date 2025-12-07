import torch
import triton
import triton.language as tl

@triton.jit
def s352_kernel(a_ptr, b_ptr, dot_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    dot_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        dot_sum += a_vals * b_vals
    
    dot_result = tl.sum(dot_sum)
    
    if tl.program_id(0) == 0:
        tl.store(dot_ptr, dot_result)

def s352_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    dot = torch.zeros(1, dtype=torch.float32, device=a.device)
    
    grid = (1,)
    
    s352_kernel[grid](
        a, b, dot,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return dot.item()