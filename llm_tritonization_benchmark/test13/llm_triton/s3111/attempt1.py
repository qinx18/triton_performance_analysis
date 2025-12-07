import torch
import triton
import triton.language as tl

@triton.jit
def s3111_kernel(a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    partial_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        condition_mask = a_vals > 0.0
        
        partial_sum += tl.where(condition_mask & mask, a_vals, 0.0)
    
    result = tl.sum(partial_sum)
    tl.store(tl.program_id(0) + tl.zeros([1], dtype=tl.int32), result)

def s3111_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create output tensor for partial sums from each block
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    partial_sums = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    s3111_kernel[(num_blocks,)](
        a,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum partial results on CPU/GPU
    total_sum = torch.sum(partial_sums).item()
    return total_sum