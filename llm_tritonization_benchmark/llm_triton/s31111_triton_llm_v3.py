import torch
import triton
import triton.language as tl

@triton.jit
def test(ptr):
    return tl.load(ptr)

@triton.jit
def s31111_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Each thread computes one sum
    if block_start < n_elements:
        sum_val = 0.0
        
        # Load values with stride of 4 elements
        for i in range(8):  # 8 calls to test function
            idx = i * 4
            if block_start == 0:  # Only first thread does the computation
                val = tl.load(a_ptr + idx)
                sum_val += val
        
        # Store result only from first thread
        if pid == 0:
            tl.store(output_ptr, sum_val)

def s31111_triton(a):
    n_elements = a.numel()
    
    # Output tensor to store the sum
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s31111_kernel[grid](
        a, output, n_elements, BLOCK_SIZE
    )
    
    return output.item()