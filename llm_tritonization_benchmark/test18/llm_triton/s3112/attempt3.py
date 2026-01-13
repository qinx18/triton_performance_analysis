import torch
import triton
import triton.language as tl

@triton.jit
def s3112_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0)
    
    if program_id == 0:
        running_sum = 0.0
        offsets = tl.arange(0, BLOCK_SIZE)
        
        for block_start in range(0, n_elements, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < n_elements
            
            vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
            
            for i in tl.static_range(BLOCK_SIZE):
                if block_start + i < n_elements:
                    running_sum += tl.load(a_ptr + block_start + i)
                    tl.store(b_ptr + block_start + i, running_sum)

def s3112_triton(a, b):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = min(256, triton.next_power_of_2(n_elements))
    
    grid = (1,)
    
    s3112_kernel[grid](
        a, b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return b[-1].item()