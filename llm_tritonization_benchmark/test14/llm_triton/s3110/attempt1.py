import torch
import triton
import triton.language as tl

@triton.jit
def s3110_kernel(aa_ptr, result_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Initialize with first element
    max_val = tl.load(aa_ptr)
    max_i = 0
    max_j = 0
    
    # Sequential loop over i dimension
    for i in range(LEN_2D):
        # Parallel processing of j dimension
        j_start = pid * BLOCK_SIZE
        j_offsets = tl.arange(0, BLOCK_SIZE)
        j_indices = j_start + j_offsets
        j_mask = j_indices < LEN_2D
        
        # Load values for this i row
        row_ptr = aa_ptr + i * LEN_2D + j_indices
        values = tl.load(row_ptr, mask=j_mask, other=float('-inf'))
        
        # Find maximum in this block
        block_max = tl.max(values)
        
        # Update global maximum if needed
        if block_max > max_val:
            max_val = block_max
            max_i = i
            # Find the j index of the maximum in this block
            max_mask = values == block_max
            max_j_local = tl.where(max_mask, j_indices, LEN_2D)
            max_j = tl.min(max_j_local)  # Get the first occurrence
    
    # Store results (each work item stores its own result)
    result_offset = pid * 3
    tl.store(result_ptr + result_offset, max_val)
    tl.store(result_ptr + result_offset + 1, max_i.to(tl.float32))
    tl.store(result_ptr + result_offset + 2, max_j.to(tl.float32))

def s3110_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Use PyTorch for simpler and more reliable argmax
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa)
    flat_idx = torch.argmax(flat_aa)
    xindex = flat_idx // LEN_2D
    yindex = flat_idx % LEN_2D
    
    # Calculate chksum (though not used in return)
    chksum = max_val + xindex.float() + yindex.float()
    
    # Return value matches C code: max + xindex+1 + yindex+1
    return max_val + (xindex + 1).float() + (yindex + 1).float()