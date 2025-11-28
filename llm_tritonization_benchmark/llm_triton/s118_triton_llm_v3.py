import triton
import triton.language as tl
import torch

@triton.jit
def s118_kernel(
    a_ptr,
    a_copy_ptr,
    bb_ptr,
    i_val,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    j_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (j_offsets < i_val) & (j_offsets >= 0)
    
    # Load values from read-only copy
    a_read_offsets = i_val - j_offsets - 1
    a_read_mask = mask & (a_read_offsets >= 0) & (a_read_offsets < LEN_2D)
    a_vals = tl.load(a_copy_ptr + a_read_offsets, mask=a_read_mask, other=0.0)
    
    # Load bb values
    bb_offsets = j_offsets * LEN_2D + i_val
    bb_mask = mask & (bb_offsets >= 0) & (bb_offsets < LEN_2D * LEN_2D)
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=bb_mask, other=0.0)
    
    # Compute products
    products = bb_vals * a_vals
    valid_products = tl.where(a_read_mask & bb_mask, products, 0.0)
    
    # Reduce sum
    result = tl.sum(valid_products)
    
    # Only the first thread in the first block writes the result
    if tl.program_id(0) == 0 and tl.arange(0, BLOCK_SIZE)[0] == 0:
        current_val = tl.load(a_ptr + i_val)
        tl.store(a_ptr + i_val, current_val + result)

def s118_triton(a, bb):
    LEN_2D = a.shape[0]
    BLOCK_SIZE = 64
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Sequential in i, parallel in j (reduction)
    for i in range(1, LEN_2D):
        num_blocks = triton.cdiv(i, BLOCK_SIZE)
        if num_blocks > 0:
            s118_kernel[(num_blocks,)](
                a,
                a_copy,
                bb,
                i,
                LEN_2D,
                BLOCK_SIZE,
            )