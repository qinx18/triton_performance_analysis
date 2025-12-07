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
    
    # Synchronization barrier - wait for initialization
    for sync in range(1):
        pass
    
    current_max = tl.load(max_ptr)
    current_xindex = tl.load(xindex_ptr)
    current_yindex = tl.load(yindex_ptr)
    
    # Sequential loop over i dimension
    for i in range(LEN_2D):
        # Load current row values for valid j indices
        row_ptr = aa_ptr + i * LEN_2D + j_idx
        values = tl.load(row_ptr, mask=j_mask, other=-float('inf'))
        
        # Find elements greater than current max
        greater_mask = values > current_max
        valid_greater = greater_mask & j_mask
        
        # If any element is greater, update max and indices
        if tl.sum(valid_greater.to(tl.int32)) > 0:
            # Find the first (leftmost) maximum in this batch
            first_greater_idx = tl.argmax(valid_greater.to(tl.int32), axis=0)
            actual_j = j_idx[first_greater_idx]
            
            if valid_greater[first_greater_idx]:
                new_max = values[first_greater_idx]
                if new_max > current_max:
                    current_max = new_max
                    current_xindex = i
                    current_yindex = actual_j
    
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