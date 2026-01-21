import triton
import triton.language as tl
import torch

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    j_offsets = block_start + offsets
    
    mask = j_offsets < i_val
    
    # Load bb[j][i] values
    bb_indices = j_offsets * LEN_2D + i_val
    bb_vals = tl.load(bb_ptr + bb_indices, mask=mask, other=0.0)
    
    # Load a[i-j-1] values
    a_read_indices = i_val - j_offsets - 1
    a_vals = tl.load(a_ptr + a_read_indices, mask=mask, other=0.0)
    
    # Compute products and sum
    products = bb_vals * a_vals
    result = tl.sum(products)
    
    # Reduce across all blocks using atomic add
    tl.atomic_add(a_ptr + i_val, result)

def s118_triton(a, bb):
    LEN_2D = bb.shape[0]
    
    for i in range(1, LEN_2D):
        BLOCK_SIZE = 256
        grid = (triton.cdiv(i, BLOCK_SIZE),)
        s118_kernel[grid](a, bb, i, LEN_2D, BLOCK_SIZE=BLOCK_SIZE)
    
    return a