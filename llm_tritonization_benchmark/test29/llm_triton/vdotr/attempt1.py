import triton
import triton.language as tl
import torch

@triton.jit
def vdotr_kernel(a_ptr, b_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize accumulator
    block_dot = 0.0
    
    # Process array in blocks
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load values
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute dot product for this block
        block_dot += tl.sum(a_vals * b_vals, axis=0)
    
    # Store result (only one thread should write)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, block_dot)

def vdotr_triton(a, b):
    n = a.shape[0]
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Use single thread block for reduction
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n))
    grid = (1,)
    
    vdotr_kernel[grid](
        a, b, result,
        n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()