import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel computes the sum reduction over specific array segments
    # Each work item handles the full computation since it's a reduction
    
    pid = tl.program_id(0)
    if pid > 0:  # Only need one thread block for this reduction
        return
    
    # Initialize sum
    total_sum = 0.0
    
    # Process 8 segments of 4 elements each: a[0:4], a[4:8], ..., a[28:32]
    for segment in range(8):
        start_idx = segment * 4
        segment_sum = 0.0
        
        # Sum 4 elements in this segment
        for i in range(4):
            idx = start_idx + i
            if idx < n_elements:
                val = tl.load(a_ptr + idx)
                segment_sum += val
        
        total_sum += segment_sum
    
    # Store the result
    tl.store(sum_ptr, total_sum)

def s31111_triton(a):
    n_elements = a.shape[0]
    
    # Create output tensor for the sum
    sum_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 128
    grid = (1,)  # Only need one thread block for reduction
    
    s31111_kernel[grid](
        a, sum_result, n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return sum_result.item()