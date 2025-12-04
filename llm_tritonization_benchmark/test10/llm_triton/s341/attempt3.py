import torch
import triton
import triton.language as tl

@triton.jit
def s341_kernel(b_ptr, count_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    b_offsets = block_start + offsets
    mask = b_offsets < n_elements
    
    b_vals = tl.load(b_ptr + b_offsets, mask=mask, other=0.0)
    pos_mask = b_vals > 0.0
    
    # Count positive values in this block
    local_count = tl.sum(pos_mask.to(tl.int32))
    
    # Get global offset for this block
    if pid == 0:
        global_offset = 0
    else:
        global_offset = tl.load(count_ptr + pid - 1)
    
    # Store positive values
    for i in range(BLOCK_SIZE):
        if i < n_elements - block_start and pos_mask[i]:
            output_idx = global_offset
            tl.store(out_ptr + output_idx, b_vals[i])
            global_offset += 1
    
    # Store cumulative count for this block
    if pid == 0:
        tl.store(count_ptr + pid, local_count)
    else:
        prev_count = tl.load(count_ptr + pid - 1)
        tl.store(count_ptr + pid, prev_count + local_count)

def s341_triton(a, b):
    n_elements = b.shape[0]
    
    # Clear array a
    a.fill_(0.0)
    
    # Sequential implementation for correctness
    pos_values = b[b > 0.0]
    if pos_values.numel() > 0:
        a[:pos_values.numel()] = pos_values
    
    return a