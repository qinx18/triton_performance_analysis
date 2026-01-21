import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    
    if block_id == 0:
        sum_val = 0.0
        
        # Call test() function for each offset: 0, 4, 8, 12, 16, 20, 24, 28
        for start_idx in range(8):
            offset = start_idx * 4
            
            # Implement test() function: sum 4 elements starting at offset
            test_sum = 0.0
            for i in range(4):
                idx = offset + i
                if idx < N:
                    val = tl.load(a_ptr + idx)
                    test_sum += val
            
            sum_val += test_sum
        
        # Store the final sum
        tl.store(result_ptr, sum_val)

def s31111_triton(a):
    N = a.shape[0]
    
    # Create output tensor for the sum result
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block
    BLOCK_SIZE = 256
    grid = (1,)
    
    s31111_kernel[grid](
        a, result, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()