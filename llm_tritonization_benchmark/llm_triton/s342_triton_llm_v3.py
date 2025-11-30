import torch
import triton
import triton.language as tl

@triton.jit
def s342_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel implements a sequential packing operation
    # Each block processes a chunk of the array sequentially
    
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Load the block of data
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load a values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Find how many positive values exist before this block
    j_offset = 0
    if block_start > 0:
        # Count positive values in previous blocks
        for prev_block in range(0, block_start, BLOCK_SIZE):
            prev_offsets = prev_block + tl.arange(0, BLOCK_SIZE)
            prev_mask = prev_offsets < tl.minimum(block_start, n_elements)
            if tl.sum(prev_mask.to(tl.int32)) > 0:
                prev_a_vals = tl.load(a_ptr + prev_offsets, mask=prev_mask)
                j_offset += tl.sum((prev_a_vals > 0.0).to(tl.int32))
    
    # Process each element in the block sequentially
    j = j_offset - 1
    for i in range(BLOCK_SIZE):
        if block_start + i < n_elements:
            offset = block_start + i
            a_val = tl.load(a_ptr + offset)
            if a_val > 0.0:
                j += 1
                b_val = tl.load(b_ptr + j)
                tl.store(a_ptr + offset, b_val)

def s342_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch kernel with single block to maintain sequential behavior
    s342_kernel[(1,)](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a