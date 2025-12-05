import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(
    aa_ptr,
    max_val_ptr,
    xindex_ptr,
    yindex_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < LEN_2D
    
    # Initialize with aa[0][0]
    max_val = tl.load(aa_ptr)
    best_i = 0
    best_j = 0
    
    # Sequential loop over i dimension
    for i in range(LEN_2D):
        # Load row i at columns j_idx
        row_ptr = aa_ptr + i * LEN_2D + j_idx
        row_vals = tl.load(row_ptr, mask=j_mask, other=-float('inf'))
        
        # Find which elements are greater than current max
        greater_mask = row_vals > max_val
        valid_greater = greater_mask & j_mask
        
        # Check if any element in this block is greater
        has_greater = tl.sum(tl.where(valid_greater, 1, 0)) > 0
        
        if has_greater:
            # Find the maximum value among valid greater elements
            candidate_max = tl.max(tl.where(valid_greater, row_vals, -float('inf')))
            
            # Update if this candidate is better
            if candidate_max > max_val:
                max_val = candidate_max
                best_i = i
                
                # Find the j index corresponding to this maximum
                max_match = (row_vals == candidate_max) & j_mask
                # Find first matching index using a different approach
                for local_j in range(BLOCK_SIZE):
                    global_j = pid * BLOCK_SIZE + local_j
                    if global_j < LEN_2D:
                        # Check if this element matches
                        element_val = tl.load(aa_ptr + i * LEN_2D + global_j)
                        if element_val == candidate_max:
                            best_j = global_j
                            break
    
    # Only thread 0 writes the result
    if pid == 0:
        tl.store(max_val_ptr, max_val)
        tl.store(xindex_ptr, best_i)
        tl.store(yindex_ptr, best_j)

def s13110_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = min(256, triton.next_power_of_2(LEN_2D))
    
    # Output tensors
    max_val = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    xindex = torch.zeros(1, dtype=torch.int32, device=aa.device)
    yindex = torch.zeros(1, dtype=torch.int32, device=aa.device)
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s13110_kernel[grid](
        aa,
        max_val,
        xindex,
        yindex,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return max_val.item() + xindex.item() + 1 + yindex.item() + 1