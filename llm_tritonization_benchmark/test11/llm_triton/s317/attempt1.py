import torch
import triton
import triton.language as tl

@triton.jit
def s317_kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:
        # Only one thread computes the product reduction
        q = 1.0
        for i in range(n_elements):
            q *= 0.99
        tl.store(output_ptr, q)

def s317_triton():
    LEN_1D = 32000
    n_elements = LEN_1D // 2
    
    # Create output tensor
    output = torch.zeros(1, dtype=torch.float32, device='cuda')
    
    # Launch kernel with single block
    BLOCK_SIZE = 1
    grid = (1,)
    
    s317_kernel[grid](
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()