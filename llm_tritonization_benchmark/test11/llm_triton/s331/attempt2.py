import torch
import triton
import triton.language as tl

@triton.jit
def s331_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid != 0:
        return
        
    j = -1
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        negative_mask = a_vals < 0.0
        
        # Find the highest index with negative value in this block
        for i in range(BLOCK_SIZE):
            global_idx = block_start + i
            if global_idx < n_elements:
                element_val = tl.load(a_ptr + global_idx)
                if element_val < 0.0:
                    j = global_idx
    
    tl.store(result_ptr, j)

def s331_triton(a):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    
    result = torch.tensor([-1], dtype=torch.int32, device=a.device)
    
    grid = (1,)
    
    s331_kernel[grid](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()