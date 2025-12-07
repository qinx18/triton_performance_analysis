import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, result_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # This is a reduction to find the global maximum and its indices
    # We need to process the entire 2D array to find the global max
    
    if pid == 0:  # Only one thread block does the work
        max_val = tl.load(aa_ptr)  # aa[0][0]
        max_i = 0
        max_j = 0
        
        # Sequential loop over i dimension
        for i in range(LEN_2D):
            # Process j dimension in blocks
            for j_start in range(0, LEN_2D, BLOCK_SIZE):
                j_offsets = tl.arange(0, BLOCK_SIZE)
                j_indices = j_start + j_offsets
                j_mask = j_indices < LEN_2D
                
                # Load values for this row
                row_ptr = aa_ptr + i * LEN_2D + j_indices
                vals = tl.load(row_ptr, mask=j_mask, other=float('-inf'))
                
                # Find max in this block
                for k in range(BLOCK_SIZE):
                    if j_start + k < LEN_2D:
                        val = tl.load(aa_ptr + i * LEN_2D + j_start + k)
                        if val > max_val:
                            max_val = val
                            max_i = i
                            max_j = j_start + k
        
        # Store results
        tl.store(result_ptr, max_val)
        tl.store(result_ptr + 1, max_i)
        tl.store(result_ptr + 2, max_j)

def s13110_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Create result tensor to store max, xindex, yindex
    result = torch.zeros(3, dtype=aa.dtype, device=aa.device)
    
    # Launch single thread block since this is a global reduction
    grid = (1,)
    
    s13110_kernel[grid](
        aa, result, 
        LEN_2D=LEN_2D, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    max_val = result[0].item()
    xindex = int(result[1].item())
    yindex = int(result[2].item())
    
    return max_val + xindex + 1 + yindex + 1