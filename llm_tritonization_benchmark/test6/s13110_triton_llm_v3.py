import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, max_out_ptr, xindex_out_ptr, yindex_out_ptr, 
                  LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Find global maximum and its indices in 2D array
    max_val = tl.load(aa_ptr)  # Initialize with aa[0][0]
    max_x = 0
    max_y = 0
    
    # Sequential scan through the 2D array
    for i in range(LEN_2D):
        row_offset = i * LEN_2D
        for j_start in range(0, LEN_2D, BLOCK_SIZE):
            offsets = tl.arange(0, BLOCK_SIZE)
            j_offsets = j_start + offsets
            mask = j_offsets < LEN_2D
            
            vals = tl.load(aa_ptr + row_offset + j_offsets, mask=mask, other=float('-inf'))
            
            # Check each value in the block
            for k in range(BLOCK_SIZE):
                if j_start + k < LEN_2D:
                    current_val = tl.load(aa_ptr + row_offset + j_start + k)
                    if current_val > max_val:
                        max_val = current_val
                        max_x = i
                        max_y = j_start + k
    
    # Store results
    tl.store(max_out_ptr, max_val)
    tl.store(xindex_out_ptr, max_x)
    tl.store(yindex_out_ptr, max_y)

def s13110_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = min(256, LEN_2D)
    
    # Output tensors
    max_out = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    xindex_out = torch.zeros(1, dtype=torch.int32, device=aa.device)
    yindex_out = torch.zeros(1, dtype=torch.int32, device=aa.device)
    
    # Launch kernel with single block since we need global reduction
    grid = (1,)
    s13110_kernel[grid](
        aa, max_out, xindex_out, yindex_out,
        LEN_2D, BLOCK_SIZE
    )
    
    max_val = max_out.item()
    xindex = xindex_out.item()
    yindex = yindex_out.item()
    chksum = max_val + float(xindex) + float(yindex)
    
    return max_val + xindex + 1 + yindex + 1