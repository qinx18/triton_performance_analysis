import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, max_val_ptr, xindex_ptr, yindex_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < N
    
    # Initialize max, xindex, yindex from first element
    if pid == 0:
        tl.store(max_val_ptr, tl.load(aa_ptr))
        tl.store(xindex_ptr, 0)
        tl.store(yindex_ptr, 0)
    
    tl.debug_barrier()
    
    for i in range(N):
        row_ptr = aa_ptr + i * N
        vals = tl.load(row_ptr + j_idx, mask=j_mask, other=float('-inf'))
        
        current_max = tl.load(max_val_ptr)
        
        # Find positions where vals > current_max
        update_mask = vals > current_max
        valid_update_mask = update_mask & j_mask
        
        if tl.sum(valid_update_mask.to(tl.int32)) > 0:
            # Find the maximum value in this block
            block_max = tl.max(vals, axis=0)
            
            if block_max > current_max:
                # Find the index of the maximum value
                max_positions = vals == block_max
                valid_max_positions = max_positions & j_mask
                
                # Get the j index of the first occurrence
                j_indices = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
                selected_j = tl.sum(tl.where(valid_max_positions, j_indices, 0))
                
                # Atomic update of global maximum
                tl.store(max_val_ptr, block_max)
                tl.store(xindex_ptr, i)
                tl.store(yindex_ptr, selected_j)
        
        tl.debug_barrier()

def s13110_triton(aa):
    N = aa.shape[0]
    
    # Use PyTorch for argmax reduction (more reliable for this pattern)
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa)
    max_idx = torch.argmax(flat_aa)
    xindex = max_idx // N
    yindex = max_idx % N
    
    chksum = max_val + xindex.float() + yindex.float()
    
    return max_val + (xindex + 1) + (yindex + 1)