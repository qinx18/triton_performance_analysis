import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, test_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load test function values (8 elements: test(&a[0]), test(&a[4]), ..., test(&a[28]))
    test_mask = offsets < 8
    test_vals = tl.load(test_ptr + offsets, mask=test_mask, other=0.0)
    
    # Sum all test values
    sum_val = tl.sum(test_vals)
    
    # Store result (first thread only)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, sum_val)

def s31111_triton(a, test):
    # Ensure inputs are contiguous
    a = a.contiguous()
    test = test.contiguous()
    
    # Create output tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Grid and block size
    BLOCK_SIZE = 32
    grid = (1,)
    
    # Launch kernel
    s31111_kernel[grid](
        a, test, result,
        a.numel(),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()