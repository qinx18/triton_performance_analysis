import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    # Sequential processing due to wavefront dependencies
    # Each thread block handles a portion of the computation
    block_id = tl.program_id(0)
    
    # Calculate which diagonal this block will process
    total_elements = (len_2d - 1) * (len_2d - 1)
    elements_per_block = BLOCK_SIZE
    
    start_idx = block_id * elements_per_block
    end_idx = tl.minimum(start_idx + elements_per_block, total_elements)
    
    # Process elements sequentially within this block
    for idx in range(start_idx, end_idx):
        # Convert linear index to (i, j) coordinates
        i = 1 + idx // (len_2d - 1)
        j = 1 + idx % (len_2d - 1)
        
        if i < len_2d and j < len_2d:
            # Load aa[i-1][j-1] and bb[i][j]
            aa_prev_offset = (i - 1) * len_2d + (j - 1)
            bb_curr_offset = i * len_2d + j
            aa_curr_offset = i * len_2d + j
            
            aa_prev_val = tl.load(aa_ptr + aa_prev_offset)
            bb_curr_val = tl.load(bb_ptr + bb_curr_offset)
            
            # Compute and store result
            result = aa_prev_val + bb_curr_val
            tl.store(aa_ptr + aa_curr_offset, result)

def s119_triton(aa, bb, len_2d):
    # Calculate number of elements to process
    total_elements = (len_2d - 1) * (len_2d - 1)
    
    # Use a conservative block size for sequential processing
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(total_elements, BLOCK_SIZE)
    
    # Launch kernel with single thread per block for sequential processing
    s119_kernel[(grid_size,)](
        aa, bb, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa