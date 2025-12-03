import triton
import triton.language as tl
import torch

@triton.jit
def s312_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=1.0)
    
    # Compute product within block using a loop
    block_prod = 1.0
    for i in range(BLOCK_SIZE):
        if i < tl.num_programs(0):
            val = tl.load(a_ptr + block_start + i, mask=(block_start + i) < n_elements, other=1.0)
            block_prod = block_prod * val
    
    tl.store(result_ptr + pid, block_prod)

def s312_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    partial_results = torch.ones(n_blocks, dtype=a.dtype, device=a.device)
    
    grid = (n_blocks,)
    s312_kernel[grid](
        a, partial_results, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    final_prod = torch.prod(partial_results)
    
    return final_prod.item()