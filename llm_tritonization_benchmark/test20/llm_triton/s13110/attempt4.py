import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, result_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Initialize with aa[0][0]
    first_val = tl.load(aa_ptr)
    current_max = first_val
    current_xindex = 0
    current_yindex = 0
    
    # Process all elements
    for flat_idx in range(pid * BLOCK_SIZE, len_2d * len_2d, BLOCK_SIZE * tl.num_programs(0)):
        offsets = flat_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < len_2d * len_2d
        
        vals = tl.load(aa_ptr + offsets, mask=mask, other=float('-inf'))
        
        for k in range(BLOCK_SIZE):
            if mask[k]:
                idx = offsets[k]
                val = vals[k]
                if val > current_max:
                    current_max = val
                    current_xindex = idx // len_2d
                    current_yindex = idx % len_2d
    
    # Store results for reduction
    result_offset = pid * 3
    tl.store(result_ptr + result_offset, current_max)
    tl.store(result_ptr + result_offset + 1, current_xindex)
    tl.store(result_ptr + result_offset + 2, current_yindex)

def s13110_triton(aa, len_2d):
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(len_2d * len_2d, BLOCK_SIZE)
    
    # Allocate result buffer for partial results
    result_buffer = torch.empty(num_blocks * 3, dtype=aa.dtype, device=aa.device)
    
    # Launch kernel
    grid = (num_blocks,)
    s13110_kernel[grid](aa, result_buffer, len_2d, BLOCK_SIZE)
    
    # Find global maximum from partial results
    partial_maxes = result_buffer[::3]
    partial_xindices = result_buffer[1::3]
    partial_yindices = result_buffer[2::3]
    
    best_block_idx = torch.argmax(partial_maxes)
    final_max = partial_maxes[best_block_idx]
    final_xindex = partial_xindices[best_block_idx]
    final_yindex = partial_yindices[best_block_idx]
    
    return final_max + final_xindex + 1 + final_yindex + 1