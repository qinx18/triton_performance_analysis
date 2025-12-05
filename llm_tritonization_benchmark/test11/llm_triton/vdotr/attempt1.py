import torch
import triton
import triton.language as tl

@triton.jit
def vdotr_kernel(a_ptr, b_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        products = a_vals * b_vals
        accumulator = tl.where(mask, accumulator + products, accumulator)
    
    dot_product = tl.sum(accumulator)
    tl.store(result_ptr, dot_product)

def vdotr_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    result = torch.zeros(1, dtype=torch.float32, device=a.device)
    
    vdotr_kernel[(1,)](
        a, b, result,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()