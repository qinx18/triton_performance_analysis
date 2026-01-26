import triton
import triton.language as tl
import torch

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    # Get program ID for the outer i loop
    i = tl.program_id(0)
    
    if i >= len_2d:
        return
    
    # Calculate initial k value: k = (i+1) * ((i+1) - 1) / 2 + (i+1)-1
    # Simplifying: k = i * (i + 1) / 2 + i = i * (i + 3) / 2
    k_start = i * (i + 1) // 2 + i
    
    # Process the inner j loop in blocks
    j_start = i
    j_end = len_2d
    
    # Calculate offsets for vectorized loads
    offsets = tl.arange(0, BLOCK_SIZE)
    
    current_k = k_start
    
    # Process j values from i to len_2d-1
    num_blocks = triton.cdiv(j_end - j_start, BLOCK_SIZE)
    for block_idx in range(num_blocks):
        j_block_start = j_start + block_idx * BLOCK_SIZE
        
        # Calculate which j values we're processing in this block
        j_offsets = j_block_start + offsets
        j_mask = j_offsets < j_end
        
        # Load bb[j][i] values for this block
        bb_indices = j_offsets * len_2d + i
        bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
        
        # Process each element in the block
        num_elements = tl.minimum(BLOCK_SIZE, j_end - j_block_start)
        for local_j in range(num_elements):
            global_j = j_block_start + local_j
            
            # Only process if within bounds
            if global_j < j_end:
                # Load current value from flat_2d_array
                current_val = tl.load(flat_2d_array_ptr + current_k)
                
                # Extract the bb value for this specific j
                bb_val = tl.load(bb_ptr + global_j * len_2d + i)
                
                # Update flat_2d_array[k]
                new_val = current_val + bb_val
                tl.store(flat_2d_array_ptr + current_k, new_val)
                
                # Update k for next iteration: k += j+1
                current_k += global_j + 1

def s141_triton(bb, flat_2d_array, len_2d):
    # Get dimensions
    N = len_2d
    
    # Define block size
    BLOCK_SIZE = 32
    
    # Calculate grid size for outer i loop
    grid = (N,)
    
    # Launch kernel with one thread per i value
    s141_kernel[grid](
        bb, flat_2d_array, N, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return flat_2d_array