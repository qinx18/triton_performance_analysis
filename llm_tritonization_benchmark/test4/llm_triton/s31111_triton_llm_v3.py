import torch
import triton
import triton.language as tl

@triton.jit
def test(ptr):
    return tl.load(ptr)

@triton.jit
def s31111_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    if pid == 0:
        # Only one thread block computes the sum
        sum_val = 0.0
        
        # Process in blocks of BLOCK_SIZE
        offsets = tl.arange(0, BLOCK_SIZE)
        
        for block_start in range(0, n_elements, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < n_elements
            
            # Load block of data
            a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
            
            # Check if we have the required indices (0, 4, 8, 12, 16, 20, 24, 28) in this block
            for i in range(BLOCK_SIZE):
                global_idx = block_start + i
                if global_idx < n_elements:
                    if global_idx in [0, 4, 8, 12, 16, 20, 24, 28]:
                        sum_val += a_vals[i]
        
        tl.store(result_ptr, sum_val)

def s31111_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 32
    
    # Create output tensor for the sum
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single thread block
    grid = (1,)
    s31111_kernel[grid](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()