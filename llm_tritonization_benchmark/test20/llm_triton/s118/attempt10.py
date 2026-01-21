import triton
import triton.language as tl
import torch

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_base = tl.program_id(0) * BLOCK_SIZE
    j_offsets = j_base + tl.arange(0, BLOCK_SIZE)
    
    j_mask = (j_offsets >= 0) & (j_offsets < i_val)
    
    bb_offsets = j_offsets * LEN_2D + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
    
    a_read_indices = i_val - j_offsets - 1
    a_read_mask = j_mask & (a_read_indices >= 0) & (a_read_indices < LEN_2D)
    a_vals = tl.load(a_ptr + a_read_indices, mask=a_read_mask, other=0.0)
    
    products = tl.where(a_read_mask, bb_vals * a_vals, 0.0)
    partial_sum = tl.sum(products)
    
    temp_offset = tl.program_id(0)
    tl.store(a_ptr + LEN_2D + temp_offset, partial_sum)

@triton.jit  
def s118_reduce_kernel(a_ptr, i_val, LEN_2D: tl.constexpr, num_blocks: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    block_offsets = tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < num_blocks
    
    partial_sums = tl.load(a_ptr + LEN_2D + block_offsets, mask=mask, other=0.0)
    total_sum = tl.sum(partial_sums)
    
    if tl.program_id(0) == 0:
        old_val = tl.load(a_ptr + i_val)
        tl.store(a_ptr + i_val, old_val + total_sum)

def s118_triton(a, bb):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 256
    
    max_blocks = triton.cdiv(LEN_2D, BLOCK_SIZE)
    extended_a = torch.cat([a, torch.zeros(max_blocks, dtype=a.dtype, device=a.device)])
    
    for i in range(1, LEN_2D):
        num_blocks = triton.cdiv(i, BLOCK_SIZE)
        
        if num_blocks > 0:
            grid = (num_blocks,)
            s118_kernel[grid](
                extended_a, bb, i, LEN_2D=LEN_2D, BLOCK_SIZE=BLOCK_SIZE
            )
            
            reduce_block_size = triton.next_power_of_2(num_blocks)
            grid = (1,)
            s118_reduce_kernel[grid](
                extended_a, i, LEN_2D=LEN_2D, num_blocks=num_blocks, BLOCK_SIZE=reduce_block_size
            )
    
    a[:] = extended_a[:LEN_2D]