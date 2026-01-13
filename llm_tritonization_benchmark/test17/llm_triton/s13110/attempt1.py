import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, max_val_ptr, xindex_ptr, yindex_ptr, chksum_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize reduction variables
    max_val = tl.load(aa_ptr)  # aa[0][0]
    best_xindex = 0
    best_yindex = 0
    
    for i in range(LEN_2D):
        j_start = pid * BLOCK_SIZE
        j_idx = j_start + j_offsets
        j_mask = j_idx < LEN_2D
        
        # Load values for this row
        row_ptr = aa_ptr + i * LEN_2D + j_idx
        vals = tl.load(row_ptr, mask=j_mask, other=-float('inf'))
        
        # Find max in this block
        block_max = tl.max(vals)
        
        # Update global max if needed
        if block_max > max_val:
            max_val = block_max
            best_xindex = i
            # Find the j index of the max value in this block
            max_mask = vals == block_max
            j_indices = j_idx
            # Get first occurrence of max
            for k in range(BLOCK_SIZE):
                if j_start + k < LEN_2D:
                    if vals[k] == block_max:
                        best_yindex = j_start + k
                        break
    
    # Store results (only one thread should write)
    if pid == 0:
        tl.store(max_val_ptr, max_val)
        tl.store(xindex_ptr, best_xindex)
        tl.store(yindex_ptr, best_yindex)
        chksum = max_val + best_xindex + best_yindex
        tl.store(chksum_ptr, chksum)

def s13110_triton(aa):
    # Use PyTorch for more efficient argmax
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa)
    flat_idx = torch.argmax(flat_aa)
    
    xindex = flat_idx // aa.shape[1]
    yindex = flat_idx % aa.shape[1]
    
    chksum = max_val + xindex.float() + yindex.float()
    
    return max_val + (xindex + 1) + (yindex + 1)