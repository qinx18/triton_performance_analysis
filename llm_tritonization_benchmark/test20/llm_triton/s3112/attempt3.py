import triton
import triton.language as tl
import torch

@triton.jit
def s3112_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n
    
    # Load previous sum from the last element of previous block
    prev_sum = 0.0
    if pid > 0:
        prev_sum = tl.load(b_ptr + block_start - 1)
    
    # Load current block data
    vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    
    # Compute cumulative sum within block
    results = tl.zeros_like(vals)
    
    for i in range(BLOCK_SIZE):
        element_mask = offsets == i
        if tl.sum(element_mask.to(tl.int32)) > 0:
            if i == 0:
                current_sum = prev_sum + tl.sum(tl.where(element_mask, vals, 0.0))
            else:
                prev_result = tl.sum(tl.where(offsets == (i-1), results, 0.0))
                current_val = tl.sum(tl.where(element_mask, vals, 0.0))
                current_sum = prev_result + current_val
            results = tl.where(element_mask, current_sum, results)
    
    tl.store(b_ptr + indices, results, mask=mask)

def s3112_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Process sequentially block by block to maintain dependencies
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    for block_id in range(num_blocks):
        grid = (1,)
        s3112_kernel[grid](
            a.data_ptr() + block_id * BLOCK_SIZE * a.element_size(),
            b.data_ptr() + block_id * BLOCK_SIZE * b.element_size(),
            min(BLOCK_SIZE, n - block_id * BLOCK_SIZE),
            BLOCK_SIZE
        )
    
    return b[n-1].item()