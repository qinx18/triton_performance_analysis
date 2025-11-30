import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_start = pid * BLOCK_SIZE
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid j values (j <= i-1)
    j_mask = j_offsets < i_val
    
    # Load bb[j][i] values
    bb_offsets = j_offsets * LEN_2D + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
    
    # Load a[i-j-1] values
    a_read_indices = i_val - j_offsets - 1
    a_read_mask = j_mask & (a_read_indices >= 0)
    a_vals = tl.load(a_ptr + a_read_indices, mask=a_read_mask, other=0.0)
    
    # Compute products and sum reduction
    products = bb_vals * a_vals
    result = tl.sum(products)
    
    # Only the first thread in the block writes back
    if pid == 0 and tl.program_id(0) == 0:
        # Load current a[i], add result, store back
        current_a = tl.load(a_ptr + i_val)
        total_sum = current_a + result
        
        # Need to accumulate from all blocks
        if pid == 0:
            tl.store(a_ptr + i_val, total_sum)

@triton.jit
def s118_reduction_kernel(a_ptr, bb_ptr, temp_ptr, i_val, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_start = pid * BLOCK_SIZE
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid j values (j <= i-1)
    j_mask = j_offsets < i_val
    
    # Load bb[j][i] values
    bb_offsets = j_offsets * LEN_2D + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
    
    # Load a[i-j-1] values
    a_read_indices = i_val - j_offsets - 1
    a_read_mask = j_mask & (a_read_indices >= 0)
    a_vals = tl.load(a_ptr + a_read_indices, mask=a_read_mask, other=0.0)
    
    # Compute products and sum within block
    products = bb_vals * a_vals
    block_sum = tl.sum(products)
    
    # Store partial sum to temporary array
    if tl.program_id(0) == 0:
        tl.store(temp_ptr + pid, block_sum)

@triton.jit 
def s118_final_kernel(a_ptr, temp_ptr, i_val, num_blocks, BLOCK_SIZE: tl.constexpr):
    # Sum all partial results
    total = 0.0
    for block_id in range(num_blocks):
        partial = tl.load(temp_ptr + block_id)
        total += partial
    
    # Add to a[i]
    current_a = tl.load(a_ptr + i_val)
    tl.store(a_ptr + i_val, current_a + total)

def s118_triton(a, bb):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 256
    
    # Create temporary storage for partial reductions
    max_blocks = triton.cdiv(LEN_2D, BLOCK_SIZE)
    temp = torch.zeros(max_blocks, device=a.device, dtype=a.dtype)
    
    # Sequential loop over i
    for i_val in range(1, LEN_2D):
        num_j = i_val  # j goes from 0 to i-1
        num_blocks = triton.cdiv(num_j, BLOCK_SIZE)
        
        if num_blocks == 1:
            # Single block case - can write directly
            grid = (1,)
            s118_kernel[grid](
                a, bb, i_val, LEN_2D, BLOCK_SIZE
            )
        else:
            # Multi-block case - need reduction
            grid = (num_blocks,)
            s118_reduction_kernel[grid](
                a, bb, temp, i_val, LEN_2D, BLOCK_SIZE
            )
            
            # Final reduction
            grid = (1,)
            s118_final_kernel[grid](
                a, temp, i_val, num_blocks, BLOCK_SIZE
            )