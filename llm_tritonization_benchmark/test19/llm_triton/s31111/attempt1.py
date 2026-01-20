import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, N, BLOCK_SIZE: tl.constexpr):
    # This kernel computes the sum reduction across specific array sections
    # Each block handles one iteration of the sum computation
    
    block_id = tl.program_id(0)
    
    if block_id == 0:
        # Compute sum for each group of 4 elements starting at offsets 0, 4, 8, 12, 16, 20, 24, 28
        total_sum = 0.0
        
        # Process 8 groups of 4 elements each
        for group_idx in range(8):
            start_offset = group_idx * 4
            group_sum = 0.0
            
            # Sum 4 elements in this group
            for i in range(4):
                offset = start_offset + i
                if offset < N:
                    val = tl.load(a_ptr + offset)
                    group_sum += val
            
            total_sum += group_sum
        
        # Store the final sum
        tl.store(sum_ptr, total_sum)

def s31111_triton(a):
    N = a.shape[0]
    
    # Output tensor for the sum
    sum_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block since we're doing a global reduction
    BLOCK_SIZE = 32
    grid = (1,)
    
    s31111_kernel[grid](
        a, sum_result, N, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return sum_result.item()