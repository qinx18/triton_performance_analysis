import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Each block processes one iteration of the sum computation
    block_id = tl.program_id(0)
    
    if block_id == 0:
        # Initialize sum
        sum_val = 0.0
        
        # Process 8 groups of 4 elements each (offsets 0, 4, 8, 12, 16, 20, 24, 28)
        for group in range(8):
            base_offset = group * 4
            group_sum = 0.0
            
            # Sum 4 elements in this group (implementing test() function)
            for i in range(4):
                offset = base_offset + i
                if offset < N:
                    val = tl.load(a_ptr + offset)
                    group_sum += val
            
            sum_val += group_sum
        
        # Store the result
        tl.store(result_ptr, sum_val)

def s31111_triton(a):
    N = a.shape[0]
    
    # Create output tensor for the sum result
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block since we're doing a global reduction
    BLOCK_SIZE = 256
    grid = (1,)
    
    s31111_kernel[grid](
        a, result, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()