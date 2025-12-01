import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Pre-define offsets once at kernel start
    base_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process j values in blocks
    block_start = pid * BLOCK_SIZE
    j_offsets = block_start + base_offsets
    
    # Mask for valid j values (j <= i_val - 1)
    j_mask = j_offsets <= (i_val - 1)
    
    # Load bb[j][i_val] values
    bb_indices = j_offsets * LEN_2D + i_val
    bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
    
    # Load a[i_val - j - 1] values
    a_indices = i_val - j_offsets - 1
    a_vals = tl.load(a_ptr + a_indices, mask=j_mask, other=0.0)
    
    # Compute products
    products = bb_vals * a_vals
    
    # Sum all valid products in this block
    block_sum = tl.sum(products)
    
    # Store the block sum to a temporary location
    # We'll use atomic add to accumulate across blocks
    temp_ptr = a_ptr + LEN_2D + pid  # Use space after array for temporary storage
    tl.store(temp_ptr, block_sum)

@triton.jit
def s118_reduce_kernel(a_ptr, temp_ptr, i_val, num_blocks: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Reduce across all blocks for this i_val
    base_offsets = tl.arange(0, BLOCK_SIZE)
    
    total_sum = 0.0
    for block_id in range(num_blocks):
        if block_id < num_blocks:
            block_sum = tl.load(temp_ptr + block_id)
            total_sum += block_sum
    
    # Add to a[i_val]
    current_val = tl.load(a_ptr + i_val)
    new_val = current_val + total_sum
    tl.store(a_ptr + i_val, new_val)

def s118_triton(a, bb):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 256
    
    # Ensure we have enough temporary space
    if a.shape[0] < LEN_2D + triton.cdiv(LEN_2D, BLOCK_SIZE):
        temp_space = torch.zeros(triton.cdiv(LEN_2D, BLOCK_SIZE), device=a.device, dtype=a.dtype)
    else:
        temp_space = None
    
    for i_val in range(1, LEN_2D):
        max_j = i_val - 1
        num_blocks = triton.cdiv(max_j + 1, BLOCK_SIZE)
        
        if num_blocks > 0:
            # Launch computation kernel
            grid = (num_blocks,)
            if temp_space is not None:
                temp_ptr = temp_space
            else:
                temp_ptr = a[LEN_2D:]
            
            s118_kernel[grid](
                a, bb, i_val, LEN_2D, BLOCK_SIZE
            )
            
            # Launch reduction kernel
            s118_reduce_kernel[(1,)](
                a, temp_ptr if temp_space is not None else a[LEN_2D:], 
                i_val, num_blocks, BLOCK_SIZE
            )