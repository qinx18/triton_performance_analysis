import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    # Initialize sum
    total_sum = 0.0
    
    # Process each segment starting at indices 0, 4, 8, 12, 16, 20, 24, 28
    for segment_start in range(8):
        start_idx = segment_start * 4
        
        # Load 4 elements from this segment
        offsets = tl.arange(0, 4)
        indices = start_idx + offsets
        mask = indices < n_elements
        
        vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
        segment_sum = tl.sum(vals)
        total_sum += segment_sum
    
    # Store the result
    tl.store(sum_ptr, total_sum)

def s31111_triton(a):
    n_elements = a.shape[0]
    
    # Create output tensor for the sum
    sum_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 128
    grid = (1,)
    
    s31111_kernel[grid](
        a, sum_result, n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return sum_result.item()