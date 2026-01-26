import triton
import triton.language as tl
import torch

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i = pid
    
    if i >= len_2d:
        return
    
    # Calculate starting k for this i
    k = (i + 1) * i // 2 + i
    
    # Process j from i to len_2d-1
    for j_start in range(i, len_2d, BLOCK_SIZE):
        j_end = min(j_start + BLOCK_SIZE, len_2d)
        actual_size = j_end - j_start
        
        if actual_size <= 0:
            break
            
        # Load bb values for current j range
        j_offsets = tl.arange(0, BLOCK_SIZE)
        j_indices = j_start + j_offsets
        j_mask = j_offsets < actual_size
        
        bb_indices = j_indices * len_2d + i
        bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
        
        # Calculate k values for current j range
        k_current = k
        for idx in range(actual_size):
            j = j_start + idx
            # Load current flat_2d_array value
            old_val = tl.load(flat_2d_array_ptr + k_current)
            # Add bb value
            new_val = old_val + tl.load(bb_ptr + j * len_2d + i)
            # Store back
            tl.store(flat_2d_array_ptr + k_current, new_val)
            # Update k for next j
            k_current += j + 1

def s141_triton(bb, flat_2d_array, len_2d):
    BLOCK_SIZE = 32
    grid = (len_2d,)
    
    s141_kernel[grid](
        bb, flat_2d_array, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )