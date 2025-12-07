import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, max_ptr, xindex_ptr, yindex_ptr, chksum_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < LEN_2D
    
    # Initialize with aa[0][0]
    if pid == 0:
        init_val = tl.load(aa_ptr)
        tl.store(max_ptr, init_val)
        tl.store(xindex_ptr, 0)
        tl.store(yindex_ptr, 0)
    
    # Load current values (these will be updated as we find better maxima)
    current_max = tl.load(max_ptr)
    current_xindex = tl.load(xindex_ptr)
    current_yindex = tl.load(yindex_ptr)
    
    # Sequential loop over i dimension
    for i in range(LEN_2D):
        # Load current row values for valid j indices
        row_ptrs = aa_ptr + i * LEN_2D + j_idx
        values = tl.load(row_ptrs, mask=j_mask, other=-float('inf'))
        
        # Find elements greater than current max
        greater_mask = values > current_max
        valid_greater = greater_mask & j_mask
        
        # Check if we found any greater elements in this block
        has_greater = tl.sum(valid_greater.to(tl.int32)) > 0
        
        if has_greater:
            # Find all locations where value > current_max
            for k in range(BLOCK_SIZE):
                if k + pid * BLOCK_SIZE < LEN_2D:
                    actual_j = k + pid * BLOCK_SIZE
                    if values[k] > current_max:
                        current_max = values[k]
                        current_xindex = i
                        current_yindex = actual_j
                        break
    
    # Store final results (only first thread writes)
    if pid == 0:
        tl.store(max_ptr, current_max)
        tl.store(xindex_ptr, current_xindex)
        tl.store(yindex_ptr, current_yindex)
        chksum = current_max + current_xindex + current_yindex
        tl.store(chksum_ptr, chksum)

def s13110_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Create output tensors
    max_val = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    xindex = torch.zeros(1, dtype=torch.int32, device=aa.device)
    yindex = torch.zeros(1, dtype=torch.int32, device=aa.device) 
    chksum = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s13110_kernel[grid](
        aa, max_val, xindex, yindex, chksum, 
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return max_val.item() + xindex.item() + 1 + yindex.item() + 1