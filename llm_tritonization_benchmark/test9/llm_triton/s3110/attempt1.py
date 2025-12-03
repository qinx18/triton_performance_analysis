import torch
import triton
import triton.language as tl

@triton.jit
def s3110_kernel(aa_ptr, max_val_ptr, max_i_ptr, max_j_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Each block processes one row
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Load current maximum values for this block
    current_max = tl.load(max_val_ptr)
    current_max_i = tl.load(max_i_ptr)
    current_max_j = tl.load(max_j_ptr)
    
    # Process row i in blocks
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_start in range(0, LEN_2D, BLOCK_SIZE):
        j_indices = j_start + j_offsets
        j_mask = j_indices < LEN_2D
        
        # Load values from row i
        row_offsets = i * LEN_2D + j_indices
        values = tl.load(aa_ptr + row_offsets, mask=j_mask, other=float('-inf'))
        
        # Check each element against current maximum
        for k in range(BLOCK_SIZE):
            if j_start + k < LEN_2D:
                val_k = tl.load(aa_ptr + i * LEN_2D + j_start + k)
                if val_k > current_max:
                    current_max = val_k
                    current_max_i = i
                    current_max_j = j_start + k
    
    # Store updated maximum values
    tl.store(max_val_ptr, current_max)
    tl.store(max_i_ptr, current_max_i)
    tl.store(max_j_ptr, current_max_j)

@triton.jit
def s3110_reduction_kernel(aa_ptr, result_ptr, LEN_2D: tl.constexpr):
    # Single block that finds global maximum
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    # Initialize with first element
    max_val = tl.load(aa_ptr)
    max_i = 0
    max_j = 0
    
    # Sequential scan through all elements
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            val = tl.load(aa_ptr + i * LEN_2D + j)
            if val > max_val:
                max_val = val
                max_i = i
                max_j = j
    
    # Store results: max_val, max_i, max_j
    tl.store(result_ptr, max_val)
    tl.store(result_ptr + 1, max_i.to(tl.float32))
    tl.store(result_ptr + 2, max_j.to(tl.float32))

def s3110_triton(aa):
    LEN_2D = aa.shape[0]
    
    # Allocate result tensor to store [max_val, max_i, max_j]
    result = torch.zeros(3, dtype=aa.dtype, device=aa.device)
    
    # Launch single-block reduction kernel
    grid = (1,)
    BLOCK_SIZE = 1024
    
    s3110_reduction_kernel[grid](
        aa,
        result,
        LEN_2D=LEN_2D
    )
    
    max_val = result[0].item()
    xindex = int(result[1].item())
    yindex = int(result[2].item())
    
    chksum = max_val + float(xindex) + float(yindex)
    
    return max_val + xindex + 1 + yindex + 1