import torch
import triton
import triton.language as tl

@triton.jit
def s3110_kernel(aa_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Handle the case where we might be the only block
    if pid == 0:
        # Initialize max, xindex, yindex with aa[0][0]
        max_val = tl.load(aa_ptr)  # aa[0][0]
        xindex = 0
        yindex = 0
        
        # Sequential loop over i (rows)
        for i in range(N):
            # Parallel processing of j values within each row
            j_offsets = tl.arange(0, BLOCK_SIZE)
            
            for j_start in range(0, N, BLOCK_SIZE):
                j_indices = j_start + j_offsets
                j_mask = j_indices < N
                
                # Load aa[i][j] values for this block
                row_ptr = aa_ptr + i * N
                values = tl.load(row_ptr + j_indices, mask=j_mask, other=float('-inf'))
                
                # Find local maximum in this block
                local_max = tl.max(values, axis=0)
                
                # Check if local max is greater than global max
                if local_max > max_val:
                    max_val = local_max
                    xindex = i
                    
                    # Find the j index of the maximum value in this block
                    max_mask = values == local_max
                    valid_mask = j_mask & max_mask
                    
                    # Get the first occurrence of the maximum
                    for idx in range(BLOCK_SIZE):
                        if j_start + idx < N and valid_mask[idx]:
                            yindex = j_start + idx
                            break
        
        # Store results
        tl.store(result_ptr, max_val)
        tl.store(result_ptr + 1, xindex.to(tl.float32))
        tl.store(result_ptr + 2, yindex.to(tl.float32))

def s3110_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Allocate result tensor to store max, xindex, yindex
    result = torch.zeros(3, dtype=aa.dtype, device=aa.device)
    
    # Launch single block to handle the reduction
    grid = (1,)
    s3110_kernel[grid](aa, result, N, BLOCK_SIZE)
    
    max_val = result[0]
    xindex = result[1]
    yindex = result[2]
    
    # Return max + xindex+1 + yindex+1 (as in C code)
    return max_val + (xindex + 1) + (yindex + 1)