import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(aa_ptr, bb_ptr, flat_2d_array_ptr, len_2d, output_size_ptr):
    # This kernel processes the entire 2D array sequentially
    # due to the stream compaction pattern requiring ordered output
    
    pid = tl.program_id(0)
    if pid != 0:  # Only use first program
        return
    
    # Initialize counter
    k = 0
    
    # Sequential processing to maintain order
    for i in range(len_2d):
        for j in range(len_2d):
            # Calculate linear index for 2D arrays (column-major: bb[j][i])
            idx = j * len_2d + i
            
            # Load condition value
            bb_val = tl.load(bb_ptr + idx)
            
            # Check condition
            if bb_val > 0.0:
                # Load corresponding aa value
                aa_val = tl.load(aa_ptr + idx)
                
                # Store in flat array at position k
                tl.store(flat_2d_array_ptr + k, aa_val)
                k += 1
    
    # Store the number of packed elements
    tl.store(output_size_ptr, k)

def s343_triton(aa, bb, flat_2d_array):
    len_2d = aa.shape[0]
    
    # Create a tensor to store the output size
    output_size = torch.zeros(1, dtype=torch.int32, device=aa.device)
    
    # Launch kernel with single program
    grid = (1,)
    s343_kernel[grid](
        aa, bb, flat_2d_array, len_2d, output_size
    )
    
    return output_size.item()