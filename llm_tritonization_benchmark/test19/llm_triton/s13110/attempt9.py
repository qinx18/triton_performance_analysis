import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, result_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid > 0:
        return
    
    # Initialize max with aa[0][0]
    max_val = tl.load(aa_ptr)
    max_i = 0
    max_j = 0
    
    # Sequential loop over i
    for i in range(len_2d):
        # Parallel processing of j values
        for j_start in range(0, len_2d, BLOCK_SIZE):
            j_offsets = tl.arange(0, BLOCK_SIZE)
            j_indices = j_start + j_offsets
            j_mask = j_indices < len_2d
            
            # Load aa[i][j] values
            row_offset = i * len_2d
            indices = row_offset + j_indices
            values = tl.load(aa_ptr + indices, mask=j_mask, other=float('-inf'))
            
            # Find maximum in this block
            block_max = tl.max(values, axis=0)
            
            # Check if block max is greater than current max
            if block_max > max_val:
                max_val = block_max
                max_i = i
                
                # Find the j index of the maximum value
                max_mask = values == block_max
                for k in range(BLOCK_SIZE):
                    if j_start + k < len_2d and (k == 0 or not max_mask[k-1]) and max_mask[k]:
                        max_j = j_start + k
                        break
    
    # Store result: max + xindex+1 + yindex+1
    result = max_val + max_i + 1 + max_j + 1
    tl.store(result_ptr, result)

def s13110_triton(aa, len_2d):
    BLOCK_SIZE = 64
    
    # Allocate result tensor
    result = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    
    # Launch kernel
    grid = (1,)
    s13110_kernel[grid](
        aa, result, len_2d, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()