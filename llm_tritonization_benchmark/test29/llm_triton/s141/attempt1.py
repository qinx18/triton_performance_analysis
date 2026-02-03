import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    # Get program ID for outer loop (i dimension)
    pid = tl.program_id(0)
    i = pid
    
    if i >= len_2d:
        return
    
    # Calculate initial k value: k = (i+1) * ((i+1) - 1) / 2 + (i+1) - 1
    i_plus_1 = i + 1
    k_base = i_plus_1 * (i_plus_1 - 1) // 2 + i_plus_1 - 1
    
    # Process inner loop in blocks
    j_start = i
    j_end = len_2d
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(j_start, j_end, BLOCK_SIZE):
        # Calculate current j values
        j_offsets = block_start + offsets
        mask = j_offsets < j_end
        
        # Load bb[j][i] values
        bb_indices = j_offsets * len_2d + i
        bb_vals = tl.load(bb_ptr + bb_indices, mask=mask, other=0.0)
        
        # Calculate k values for this block
        # k = k_base + sum(j+1 for j in range(j_start, block_start))
        k_offset = 0
        for j in range(j_start, block_start):
            k_offset += j + 1
        
        # For each j in the block, calculate its k position and update
        for local_idx in range(BLOCK_SIZE):
            if block_start + local_idx < j_end:
                j = block_start + local_idx
                k = k_base + k_offset
                
                # Update k_offset for next iteration
                if local_idx > 0:
                    k_offset += j
                
                # Load current value, add bb value, store back
                current_val = tl.load(flat_2d_array_ptr + k)
                bb_val = tl.load(bb_ptr + j * len_2d + i)
                new_val = current_val + bb_val
                tl.store(flat_2d_array_ptr + k, new_val)
                
                # Update k_offset for next j
                k_offset += 1

@triton.jit
def s141_kernel_sequential(bb_ptr, flat_2d_array_ptr, len_2d):
    # Sequential implementation that matches C code exactly
    for i in range(len_2d):
        # k = (i+1) * ((i+1) - 1) / 2 + (i+1) - 1
        i_plus_1 = i + 1
        k = i_plus_1 * (i_plus_1 - 1) // 2 + i_plus_1 - 1
        
        for j in range(i, len_2d):
            # flat_2d_array[k] += bb[j][i]
            bb_val = tl.load(bb_ptr + j * len_2d + i)
            current_val = tl.load(flat_2d_array_ptr + k)
            new_val = current_val + bb_val
            tl.store(flat_2d_array_ptr + k, new_val)
            
            # k += j + 1
            k += j + 1

def s141_triton(bb, flat_2d_array, len_2d):
    # Use sequential kernel due to complex dependency pattern
    grid = (1,)
    s141_kernel_sequential[grid](bb, flat_2d_array, len_2d)