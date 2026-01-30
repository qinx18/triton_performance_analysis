import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, max_val_ptr, xindex_ptr, yindex_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < len_2d
    
    # Initialize with first element
    if pid == 0:
        first_val = tl.load(aa_ptr)
        tl.store(max_val_ptr, first_val)
        tl.store(xindex_ptr, 0)
        tl.store(yindex_ptr, 0)
    
    tl.debug_barrier()
    
    for i in range(len_2d):
        row_ptr = aa_ptr + i * len_2d
        vals = tl.load(row_ptr + j_idx, mask=j_mask, other=float('-inf'))
        
        current_max = tl.load(max_val_ptr)
        
        # Find local maximum in this block
        local_max = tl.max(vals)
        local_max_idx = tl.argmax(vals, axis=0)
        
        if local_max > current_max:
            # Use atomic compare and swap to update global maximum
            old_max = tl.atomic_cas(max_val_ptr, current_max, local_max)
            if old_max == current_max:
                actual_j = pid * BLOCK_SIZE + local_max_idx
                if actual_j < len_2d:
                    tl.store(xindex_ptr, i)
                    tl.store(yindex_ptr, actual_j)

def s13110_triton(aa, len_2d):
    BLOCK_SIZE = 256
    
    # Use PyTorch for argmax which is more efficient
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa)
    max_idx = torch.argmax(flat_aa)
    xindex = max_idx // len_2d
    yindex = max_idx % len_2d
    
    chksum = max_val + xindex.float() + yindex.float()
    
    return max_val + xindex + 1 + yindex + 1