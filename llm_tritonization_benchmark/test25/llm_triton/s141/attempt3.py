import triton
import triton.language as tl
import torch

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, len_2d, i, BLOCK_SIZE: tl.constexpr):
    # Calculate initial k value: k = (i+1) * ((i+1) - 1) / 2 + (i+1)-1
    # Simplifying: k = i * (i + 1) / 2 + i = i * (i + 3) / 2
    k_start = i * (i + 1) // 2 + i
    
    # Process the inner j loop in blocks
    j_start = i
    j_end = len_2d
    
    # Calculate offsets for vectorized operations
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Get program ID for block processing
    block_idx = tl.program_id(0)
    j_block_start = j_start + block_idx * BLOCK_SIZE
    
    # Calculate which j values we're processing in this block
    j_indices = j_block_start + offsets
    j_mask = (j_indices < j_end) & (j_indices >= j_start)
    
    # Calculate k values for each j in this block
    # k starts at k_start and increments by j+1 for each step
    k_values = k_start
    for idx in range(block_idx * BLOCK_SIZE):
        if j_start + idx < j_end:
            k_values += (j_start + idx) + 1
    
    # Process valid elements in this block
    for local_idx in range(BLOCK_SIZE):
        j_idx = j_block_start + local_idx
        if j_idx < j_end and j_idx >= j_start:
            # Load bb[j][i]
            bb_idx = j_idx * len_2d + i
            bb_val = tl.load(bb_ptr + bb_idx)
            
            # Calculate current k
            current_k = k_start
            for temp_j in range(j_start, j_idx):
                current_k += temp_j + 1
            
            # Load current value from flat_2d_array
            current_val = tl.load(flat_2d_array_ptr + current_k)
            
            # Update flat_2d_array[k]
            new_val = current_val + bb_val
            tl.store(flat_2d_array_ptr + current_k, new_val)

@triton.jit
def s141_simple_kernel(bb_ptr, flat_2d_array_ptr, len_2d):
    # Get program ID for the outer i loop
    i = tl.program_id(0)
    
    if i >= len_2d:
        return
    
    # Calculate initial k value: k = (i+1) * ((i+1) - 1) / 2 + (i+1)-1
    k_start = i * (i + 1) // 2 + i
    current_k = k_start
    
    # Process j values from i to len_2d-1 sequentially
    for j in range(i, len_2d):
        # Load bb[j][i]
        bb_idx = j * len_2d + i
        bb_val = tl.load(bb_ptr + bb_idx)
        
        # Load current value from flat_2d_array
        current_val = tl.load(flat_2d_array_ptr + current_k)
        
        # Update flat_2d_array[k]
        new_val = current_val + bb_val
        tl.store(flat_2d_array_ptr + current_k, new_val)
        
        # Update k for next iteration: k += j+1
        current_k += j + 1

def s141_triton(bb, flat_2d_array, len_2d):
    # Get dimensions
    N = len_2d
    
    # Calculate grid size for outer i loop
    grid = (N,)
    
    # Launch kernel with one thread per i value
    s141_simple_kernel[grid](
        bb, flat_2d_array, N
    )
    
    return flat_2d_array