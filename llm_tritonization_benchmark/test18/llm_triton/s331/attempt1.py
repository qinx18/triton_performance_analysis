import triton
import triton.language as tl
import torch

@triton.jit
def s331_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Find the last index where a[i] < 0
    offsets = tl.arange(0, BLOCK_SIZE)
    j = -1
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=1.0)  # Use 1.0 as other to avoid false matches
        
        # Check which elements are < 0
        neg_mask = vals < 0.0
        
        # Update j for each negative element in order
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                offset_i = block_start + i
                val = tl.load(a_ptr + offset_i)
                if val < 0.0:
                    j = offset_i
    
    # Store result
    tl.store(result_ptr, j)

def s331_triton(a):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create result tensor
    result = torch.tensor(-1, dtype=torch.int32, device=a.device)
    
    # Launch kernel with single thread since we need sequential processing
    s331_kernel[(1,)](
        a, result, N, BLOCK_SIZE
    )
    
    return result.item() + 1