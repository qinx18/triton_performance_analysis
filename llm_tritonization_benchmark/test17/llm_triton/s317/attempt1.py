import torch
import triton
import triton.language as tl

@triton.jit
def s317_kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:  # Only one thread computes the product
        q = 1.0
        for i in range(n_elements):
            q *= 0.99
        tl.store(output_ptr, q)

def s317_triton():
    n_elements = 32000 // 2  # LEN_1D/2
    
    # Output tensor for result
    result = torch.zeros(1, dtype=torch.float32, device='cuda')
    
    # Launch kernel with single block
    grid = (1,)
    BLOCK_SIZE = 128
    
    s317_kernel[grid](
        result,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()