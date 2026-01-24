import triton
import triton.language as tl
import torch

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i, LEN_2D, BLOCK_SIZE: tl.constexpr):
    # Get program ID for j dimension
    pid = tl.program_id(0)
    
    # Calculate j offsets for this block
    j_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid j values (j <= i-1)
    j_mask = j_offsets < i
    
    # Load bb[j][i] values
    bb_offsets = j_offsets * LEN_2D + i
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
    
    # Calculate a[i-j-1] indices
    a_read_indices = i - j_offsets - 1
    a_read_mask = j_mask & (a_read_indices >= 0)
    
    # Load a[i-j-1] values
    a_vals = tl.load(a_ptr + a_read_indices, mask=a_read_mask, other=0.0)
    
    # Compute products
    products = bb_vals * a_vals
    
    # Mask out invalid products
    valid_products = tl.where(a_read_mask, products, 0.0)
    
    # Sum across the block
    result = tl.sum(valid_products)
    
    # Atomic add to a[i]
    if pid == 0:
        # Load current value of a[i]
        current_val = tl.load(a_ptr + i)
        # Store updated value
        tl.store(a_ptr + i, current_val + result)
    else:
        # For other blocks, use atomic add
        tl.atomic_add(a_ptr + i, result)

def s118_triton(a, bb):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 128
    
    # Ensure tensors are contiguous
    a = a.contiguous()
    bb = bb.contiguous()
    
    for i in range(1, LEN_2D):
        # Calculate number of j values for this i
        num_j = i
        
        if num_j > 0:
            # Calculate grid size
            grid_size = triton.cdiv(num_j, BLOCK_SIZE)
            
            # Launch kernel
            s118_kernel[(grid_size,)](
                a, bb, i, LEN_2D, BLOCK_SIZE
            )