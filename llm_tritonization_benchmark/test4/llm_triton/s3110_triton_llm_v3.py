import triton
import triton.language as tl
import torch

@triton.jit
def s3110_kernel(aa_ptr, max_ptr, xindex_ptr, yindex_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Initialize with aa[0][0]
    first_val = tl.load(aa_ptr)
    current_max = first_val
    current_xindex = 0
    current_yindex = 0
    
    # Define offsets once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process all elements in blocks
    total_elements = LEN_2D * LEN_2D
    for block_start in range(0, total_elements, BLOCK_SIZE):
        # Calculate current offsets
        current_offsets = block_start + offsets
        mask = current_offsets < total_elements
        
        # Load values
        vals = tl.load(aa_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Check each element in the block
        for k in range(BLOCK_SIZE):
            if block_start + k < total_elements:
                val = tl.load(aa_ptr + block_start + k)
                if val > current_max:
                    current_max = val
                    linear_idx = block_start + k
                    current_xindex = linear_idx // LEN_2D
                    current_yindex = linear_idx % LEN_2D
    
    # Store results
    if pid == 0:
        tl.store(max_ptr, current_max)
        tl.store(xindex_ptr, current_xindex)
        tl.store(yindex_ptr, current_yindex)

def s3110_triton(aa):
    LEN_2D = aa.shape[0]
    
    # Allocate output tensors
    max_result = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    xindex_result = torch.zeros(1, dtype=torch.int32, device=aa.device)
    yindex_result = torch.zeros(1, dtype=torch.int32, device=aa.device)
    
    # Launch kernel with single program
    BLOCK_SIZE = 256
    grid = (1,)
    
    s3110_kernel[grid](
        aa, max_result, xindex_result, yindex_result,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    max_val = max_result.item()
    xindex = xindex_result.item()
    yindex = yindex_result.item()
    chksum = max_val + float(xindex) + float(yindex)
    
    return max_val + xindex + 1 + yindex + 1