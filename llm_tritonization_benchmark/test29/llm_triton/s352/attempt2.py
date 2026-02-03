import triton
import triton.language as tl
import torch

@triton.jit
def s352_kernel(a_ptr, b_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    dot_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        current_mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=current_mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=current_mask, other=0.0)
        
        dot_sum += a_vals * b_vals
    
    total = tl.sum(dot_sum)
    
    pid = tl.program_id(0)
    first_thread = tl.arange(0, BLOCK_SIZE) == 0
    if pid == 0:
        tl.store(result_ptr, total, mask=first_thread)

def s352_triton(a, b):
    n_elements = a.shape[0]
    
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s352_kernel[grid](
        a, b, result,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()