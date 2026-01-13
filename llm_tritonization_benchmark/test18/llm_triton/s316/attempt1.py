import triton
import triton.language as tl
import torch

@triton.jit
def s316_kernel(a_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Calculate number of blocks needed
    num_blocks = tl.cdiv(N, BLOCK_SIZE)
    
    # Initialize with first element
    global_min = tl.load(a_ptr)
    
    # Process all blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    for block_start in range(0, N, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < N
        
        # Load block of data, use inf for masked elements so they don't affect min
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('inf'))
        
        # Find minimum in this block
        block_min = tl.min(vals)
        
        # Update global minimum
        global_min = tl.minimum(global_min, block_min)
    
    # Store result
    tl.store(result_ptr, global_min)

def s316_triton(a):
    N = a.shape[0]
    
    # Create output tensor
    result = torch.empty(1, dtype=a.dtype, device=a.device)
    
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Launch kernel with single thread block since we need global reduction
    s316_kernel[(1,)](
        a, result, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()