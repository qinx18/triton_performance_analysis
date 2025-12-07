import torch
import triton
import triton.language as tl

@triton.jit
def s3110_kernel(aa_ptr, result_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Initialize reduction variables
    max_val = tl.load(aa_ptr)  # aa[0][0]
    max_i = 0
    max_j = 0
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(LEN_2D):
        for j_block_start in range(0, LEN_2D, BLOCK_SIZE):
            j_indices = j_block_start + j_offsets
            j_mask = j_indices < LEN_2D
            
            # Calculate linear indices for aa[i][j]
            linear_indices = i * LEN_2D + j_indices
            
            # Load values from aa
            vals = tl.load(aa_ptr + linear_indices, mask=j_mask, other=float('-inf'))
            
            # Find maximum within this block
            block_max = tl.max(vals, axis=0)
            
            # Check if this block's max is greater than global max
            if block_max > max_val:
                # Find the position of the maximum within the block
                max_mask = vals == block_max
                # Get the first occurrence
                for k in range(BLOCK_SIZE):
                    if j_block_start + k < LEN_2D and vals[k] == block_max:
                        max_val = block_max
                        max_i = i
                        max_j = j_block_start + k
                        break
    
    # Only thread 0 writes the result
    if pid == 0:
        chksum = max_val + max_i + max_j
        tl.store(result_ptr, max_val)
        tl.store(result_ptr + 1, max_i)
        tl.store(result_ptr + 2, max_j)
        tl.store(result_ptr + 3, chksum)

def s3110_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 32
    
    # Create result tensor to store max, xindex, yindex, chksum
    result = torch.zeros(4, dtype=aa.dtype, device=aa.device)
    
    grid = (1,)  # Single thread group for reduction
    
    s3110_kernel[grid](
        aa, result, LEN_2D, BLOCK_SIZE
    )
    
    max_val = result[0].item()
    xindex = int(result[1].item())
    yindex = int(result[2].item())
    
    return max_val + xindex + 1 + yindex + 1