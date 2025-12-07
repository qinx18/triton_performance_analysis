import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, result_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < LEN_2D
    
    # Initialize with aa[0][0]
    init_val = tl.load(aa_ptr)
    local_max = init_val
    local_xindex = 0
    local_yindex = 0
    
    # Sequential loop over i dimension
    for i in range(LEN_2D):
        # Load current row values for valid j indices
        row_ptrs = aa_ptr + i * LEN_2D + j_idx
        values = tl.load(row_ptrs, mask=j_mask, other=-float('inf'))
        
        # Find elements greater than current max
        greater_mask = values > local_max
        valid_greater = greater_mask & j_mask
        
        # Check if we found any greater elements in this block
        has_greater_int = tl.sum(valid_greater.to(tl.int32))
        
        # Update max if we found greater elements
        max_val_in_block = tl.max(tl.where(valid_greater, values, -float('inf')))
        if max_val_in_block > local_max:
            # Find the index of the maximum element
            max_mask = (values == max_val_in_block) & valid_greater
            # Get the first occurrence
            for k in tl.static_range(BLOCK_SIZE):
                curr_j = pid * BLOCK_SIZE + k
                if curr_j < LEN_2D:
                    if max_mask[k]:
                        local_max = values[k]
                        local_xindex = i
                        local_yindex = curr_j
                        break
    
    # Store results for this block
    block_offset = pid * 3
    tl.store(result_ptr + block_offset, local_max)
    tl.store(result_ptr + block_offset + 1, local_xindex.to(tl.float32))
    tl.store(result_ptr + block_offset + 2, local_yindex.to(tl.float32))

def s13110_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    num_blocks = triton.cdiv(LEN_2D, BLOCK_SIZE)
    # Store max, xindex, yindex for each block
    result_buffer = torch.zeros(num_blocks * 3, dtype=aa.dtype, device=aa.device)
    
    grid = (num_blocks,)
    
    s13110_kernel[grid](
        aa, result_buffer,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Find global maximum among all blocks
    max_vals = result_buffer[::3]
    x_indices = result_buffer[1::3]
    y_indices = result_buffer[2::3]
    
    global_max_idx = torch.argmax(max_vals)
    final_max = max_vals[global_max_idx].item()
    final_xindex = int(x_indices[global_max_idx].item())
    final_yindex = int(y_indices[global_max_idx].item())
    
    return final_max + final_xindex + 1 + final_yindex + 1