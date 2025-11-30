import triton
import triton.language as tl

@triton.jit
def s118_kernel(
    a_ptr,
    a_copy_ptr,
    bb_ptr,
    i,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < i
    
    # Load values for j indices
    j_vals = tl.load(a_copy_ptr + offsets, mask=mask)
    
    # Load bb values: bb[j][i]
    bb_offsets = offsets * LEN_2D + i
    bb_mask = mask
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=bb_mask)
    
    # Load a values: a[i-j-1]
    a_read_offsets = i - offsets - 1
    a_read_mask = mask & (a_read_offsets >= 0)
    a_read_vals = tl.load(a_copy_ptr + a_read_offsets, mask=a_read_mask, other=0.0)
    
    # Compute partial products
    products = bb_vals * a_read_vals
    products = tl.where(mask, products, 0.0)
    
    # Sum all products in this block
    block_sum = tl.sum(products)
    
    # Store partial sum (will be accumulated across blocks)
    if tl.program_id(0) == 0:
        tl.atomic_add(a_ptr + i, block_sum)
    else:
        tl.atomic_add(a_ptr + i, block_sum)

def s118_triton(a, bb):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 256
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    for i in range(1, LEN_2D):
        if i > 0:  # Only process if there are j values to iterate over
            num_blocks = triton.cdiv(i, BLOCK_SIZE)
            grid = (num_blocks,)
            
            s118_kernel[grid](
                a,
                a_copy,
                bb,
                i,
                LEN_2D,
                BLOCK_SIZE,
            )