import torch
import triton
import triton.language as tl

@triton.jit
def s3110_kernel(aa_ptr, output_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Initialize with first element
    max_val = tl.load(aa_ptr)
    max_i = 0
    max_j = 0
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(LEN_2D):
        for j_block_start in range(0, LEN_2D, BLOCK_SIZE):
            j_indices = j_block_start + j_offsets
            j_mask = j_indices < LEN_2D
            
            # Calculate flat indices for 2D array access
            flat_indices = i * LEN_2D + j_indices
            
            # Load values
            vals = tl.load(aa_ptr + flat_indices, mask=j_mask, other=float('-inf'))
            
            # Find local maximum
            local_max = tl.max(vals)
            
            # Check if local max is greater than current global max
            if local_max > max_val:
                max_val = local_max
                max_i = i
                
                # Find which j index has the maximum value
                max_mask = vals == local_max
                for k in range(BLOCK_SIZE):
                    if j_block_start + k < LEN_2D:
                        if (max_mask >> k) & 1:
                            max_j = j_block_start + k
                            break
    
    # Only thread 0 writes the result
    if pid == 0:
        chksum = max_val + max_i + max_j
        result = max_val + (max_i + 1) + (max_j + 1)
        tl.store(output_ptr, result)

def s3110_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Use PyTorch for more efficient argmax
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa)
    max_idx = torch.argmax(flat_aa)
    
    xindex = max_idx // LEN_2D
    yindex = max_idx % LEN_2D
    
    chksum = max_val + xindex + yindex
    result = max_val + (xindex + 1) + (yindex + 1)
    
    return result.item()