import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, BLOCK_SIZE: tl.constexpr):
    j_offset = tl.program_id(0) * BLOCK_SIZE
    j_offsets = j_offset + tl.arange(0, BLOCK_SIZE)
    
    j_mask = j_offsets < i_val
    
    bb_offsets = j_offsets * 256 + i_val
    a_read_offsets = i_val - j_offsets - 1
    
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
    a_vals = tl.load(a_ptr + a_read_offsets, mask=j_mask, other=0.0)
    
    products = bb_vals * a_vals
    reduction = tl.sum(products)
    
    if tl.program_id(0) == 0:
        current_a = tl.load(a_ptr + i_val)
        tl.store(a_ptr + i_val, current_a + reduction)

@triton.jit
def s118_partial_kernel(a_ptr, bb_ptr, temp_ptr, i_val, BLOCK_SIZE: tl.constexpr):
    j_offset = tl.program_id(0) * BLOCK_SIZE
    j_offsets = j_offset + tl.arange(0, BLOCK_SIZE)
    
    j_mask = j_offsets < i_val
    
    bb_offsets = j_offsets * 256 + i_val
    a_read_offsets = i_val - j_offsets - 1
    
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
    a_vals = tl.load(a_ptr + a_read_offsets, mask=j_mask, other=0.0)
    
    products = bb_vals * a_vals
    reduction = tl.sum(products)
    
    tl.store(temp_ptr + tl.program_id(0), reduction)

@triton.jit  
def s118_reduce_kernel(temp_ptr, a_ptr, i_val, num_blocks, BLOCK_SIZE: tl.constexpr):
    block_offsets = tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < num_blocks
    
    partial_sums = tl.load(temp_ptr + block_offsets, mask=mask, other=0.0)
    total_sum = tl.sum(partial_sums)
    
    current_a = tl.load(a_ptr + i_val)
    tl.store(a_ptr + i_val, current_a + total_sum)

def s118_triton(a, bb):
    N = bb.shape[0]
    BLOCK_SIZE = 256
    
    max_blocks = triton.cdiv(N, BLOCK_SIZE)
    temp_buffer = torch.zeros(max_blocks, device=a.device, dtype=a.dtype)
    
    for i in range(1, N):
        num_blocks = triton.cdiv(i, BLOCK_SIZE)
        
        if num_blocks == 1:
            grid = (1,)
            s118_kernel[grid](a, bb, i, BLOCK_SIZE=BLOCK_SIZE)
        else:
            grid = (num_blocks,)
            s118_partial_kernel[grid](a, bb, temp_buffer, i, BLOCK_SIZE=BLOCK_SIZE)
            
            reduce_grid = (1,)
            s118_reduce_kernel[reduce_grid](temp_buffer, a, i, num_blocks, BLOCK_SIZE=BLOCK_SIZE)