import torch
import triton
import triton.language as tl

@triton.jit
def s315_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    max_val = tl.load(a_ptr)
    max_idx = 0
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                current_val = vals[i]
                if current_val > max_val:
                    max_val = current_val
                    max_idx = block_start + i
    
    result = max_idx + max_val + 1
    tl.store(result_ptr, result)

def s315_triton(a):
    n_elements = a.shape[0]
    
    result = torch.zeros(1, dtype=torch.float32, device=a.device)
    
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s315_kernel[grid](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()