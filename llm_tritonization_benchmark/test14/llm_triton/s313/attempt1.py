import torch
import triton
import triton.language as tl

@triton.jit
def s313_kernel(a_ptr, b_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    dot_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        dot_sum += a_vals * b_vals
    
    # Reduce within block
    total = tl.sum(dot_sum)
    
    # Only first thread writes result
    if tl.program_id(0) == 0:
        tl.store(result_ptr, total)

def s313_triton(a, b):
    n_elements = a.shape[0]
    result = torch.zeros(1, device=a.device, dtype=a.dtype)
    
    BLOCK_SIZE = 1024
    
    s313_kernel[(1,)](
        a, b, result,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()