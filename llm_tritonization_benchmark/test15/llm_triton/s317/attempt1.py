import triton
import triton.language as tl
import torch

@triton.jit
def s317_kernel(
    result_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid == 0:
        # Only one thread block performs the computation
        q = 1.0
        factor = 0.99
        
        # Perform the product reduction sequentially
        for i in range(n_elements):
            q *= factor
        
        # Store the result
        tl.store(result_ptr, q)

def s317_triton():
    LEN_1D = 32000
    n_elements = LEN_1D // 2
    
    # Create output tensor
    result = torch.zeros(1, dtype=torch.float32, device='cuda')
    
    BLOCK_SIZE = 128
    grid = (1,)
    
    s317_kernel[grid](
        result,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return result.item()