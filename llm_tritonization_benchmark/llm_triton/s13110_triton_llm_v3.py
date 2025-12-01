import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, max_out_ptr, xindex_out_ptr, yindex_out_ptr, chksum_out_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Initialize max, xindex, yindex with first element
    max_val = tl.load(aa_ptr)  # aa[0][0]
    xindex = 0
    yindex = 0
    
    # Iterate through all elements
    for i in range(LEN_2D):
        for j_start in range(0, LEN_2D, BLOCK_SIZE):
            offsets = tl.arange(0, BLOCK_SIZE)
            j_offsets = j_start + offsets
            mask = j_offsets < LEN_2D
            
            # Load current row values
            row_ptr = aa_ptr + i * LEN_2D + j_offsets
            vals = tl.load(row_ptr, mask=mask, other=-float('inf'))
            
            # Check each value in the block
            for block_idx in range(BLOCK_SIZE):
                if j_start + block_idx < LEN_2D:
                    current_val = tl.load(aa_ptr + i * LEN_2D + j_start + block_idx)
                    if current_val > max_val:
                        max_val = current_val
                        xindex = i
                        yindex = j_start + block_idx
    
    # Calculate checksum
    chksum = max_val + xindex + yindex
    
    # Store results
    tl.store(max_out_ptr, max_val)
    tl.store(xindex_out_ptr, xindex)
    tl.store(yindex_out_ptr, yindex)
    tl.store(chksum_out_ptr, chksum)

def s13110_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = min(256, LEN_2D)
    
    # Output tensors
    max_out = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    xindex_out = torch.zeros(1, dtype=torch.int32, device=aa.device)
    yindex_out = torch.zeros(1, dtype=torch.int32, device=aa.device)
    chksum_out = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    
    # Launch kernel
    s13110_kernel[(1,)](
        aa, max_out, xindex_out, yindex_out, chksum_out,
        LEN_2D, BLOCK_SIZE
    )
    
    return max_out.item(), xindex_out.item(), yindex_out.item(), chksum_out.item()