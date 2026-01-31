import triton
import triton.language as tl
import torch

@triton.jit
def s317_kernel(output_ptr, iterations, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:
        # Only one thread computes the product reduction
        q = 1.0
        for i in range(iterations):
            q = q * 0.99
        
        # Store result
        tl.store(output_ptr, q)

def s317_triton(n):
    # Create output tensor
    output = torch.zeros(1, dtype=torch.float32, device='cuda')
    
    # Launch kernel with single block
    grid = (1,)
    BLOCK_SIZE = 256
    
    iterations = n // 2
    
    s317_kernel[grid](
        output,
        iterations,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()