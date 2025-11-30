import triton
import triton.language as tl
import torch

@triton.jit
def s141_kernel(flat_2d_array_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Calculate initial k value: k = (i+1) * ((i+1) - 1) / 2 + (i+1) - 1
    # Simplifying: k = i * (i + 1) / 2 + i = i * (i + 3) / 2
    k = i * (i + 1) // 2 + i
    
    # Process j from i to LEN_2D-1
    j_start = i
    j_end = LEN_2D
    
    # Process in blocks
    for j_block_start in range(j_start, j_end, BLOCK_SIZE):
        j_block_end = min(j_block_start + BLOCK_SIZE, j_end)
        block_size = j_block_end - j_block_start
        
        # Load bb values for this block
        j_offsets = tl.arange(0, BLOCK_SIZE)
        j_indices = j_block_start + j_offsets
        j_mask = j_offsets < block_size
        
        bb_indices = j_indices * LEN_2D + i
        bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
        
        # Update flat_2d_array and k for each j in the block
        current_k = k
        for idx in range(block_size):
            j = j_block_start + idx
            # Load current value
            flat_val = tl.load(flat_2d_array_ptr + current_k)
            # Update with bb value
            updated_val = flat_val + bb_vals[idx]
            # Store back
            tl.store(flat_2d_array_ptr + current_k, updated_val)
            # Update k for next iteration: k += j+1
            current_k += j + 1
        
        # Update k for next block
        k = current_k

def s141_triton(flat_2d_array, bb):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 32
    
    # Launch kernel with one thread per i
    grid = (LEN_2D,)
    
    s141_kernel[grid](
        flat_2d_array,
        bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )