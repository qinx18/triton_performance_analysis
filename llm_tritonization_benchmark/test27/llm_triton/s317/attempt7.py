import triton
import triton.language as tl
import torch

@triton.jit
def s317_kernel(result_ptr):
    q = 1.0
    for i in range(16000):  # This will be n//2 iterations
        q *= 0.99
    tl.store(result_ptr, q)

def s317_triton(n):
    result = torch.zeros(1, dtype=torch.float32, device='cuda')
    
    # Launch kernel with single thread
    s317_kernel[(1,)](result)
    
    # Update the kernel to use actual n//2
    @triton.jit
    def s317_kernel_actual(result_ptr, num_iters):
        q = 1.0
        for i in range(num_iters):
            q *= 0.99
        tl.store(result_ptr, q)
    
    # Relaunch with correct iteration count
    s317_kernel_actual[(1,)](result, n // 2)
    
    return result.item()