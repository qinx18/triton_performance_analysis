import triton
import triton.language as tl
import torch

@triton.jit
def s313_kernel(a_ptr, b_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block start offset
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate global offsets
    indices = block_start + offsets
    
    # Create mask for valid elements
    mask = indices < N
    
    # Load values with masking
    a_vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Compute element-wise product and sum
    block_dot = tl.sum(a_vals * b_vals, axis=0)
    
    # Store the partial sum
    tl.store(output_ptr + pid, block_dot)

def s313_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate number of blocks
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    s313_kernel[grid](
        a, b, partial_sums, 
        N, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial results to get final dot product
    dot = torch.sum(partial_sums)
    
    return dot.item()