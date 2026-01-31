import triton
import triton.language as tl
import torch

@triton.jit
def s352_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    dot_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        current_mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=current_mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=current_mask, other=0.0)
        
        dot_sum += a_vals * b_vals
    
    # Reduce within block
    total = tl.sum(dot_sum)
    
    # Only first thread in block writes result
    if tl.program_id(0) == 0 and offsets[0] == 0:
        tl.store(output_ptr, total)

def s352_triton(a, b):
    n_elements = a.shape[0]
    
    # Output tensor for dot product result
    output = torch.zeros(1, dtype=torch.float32, device=a.device)
    
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s352_kernel[grid](
        a, b, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()