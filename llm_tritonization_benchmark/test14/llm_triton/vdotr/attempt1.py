import torch
import triton
import triton.language as tl

@triton.jit
def vdotr_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    dot_accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        dot_accumulator += a_vals * b_vals
    
    dot_result = tl.sum(dot_accumulator)
    tl.store(output_ptr, dot_result)

def vdotr_triton(a, b):
    n_elements = a.shape[0]
    output = torch.zeros(1, dtype=torch.float32, device=a.device)
    
    BLOCK_SIZE = 1024
    
    vdotr_kernel[(1,)](
        a, b, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()