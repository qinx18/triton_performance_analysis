import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for j dimension
    pid = tl.program_id(0)
    j_offset = pid * BLOCK_SIZE
    j_offsets = j_offset + tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Load a[i-1] (scalar)
    a_prev = tl.load(a_ptr + (i_val - 1))
    
    # Load aa[j][i] values
    aa_ptrs = aa_ptr + j_offsets * LEN_2D + i_val
    aa_vals = tl.load(aa_ptrs, mask=j_mask)
    
    # Load bb[j][i] values
    bb_ptrs = bb_ptr + j_offsets * LEN_2D + i_val
    bb_vals = tl.load(bb_ptrs, mask=j_mask)
    
    # Compute a[i] = aa[j][i] - a[i-1] for all j
    a_new_vals = aa_vals - a_prev
    
    # The last valid j iteration will determine a[i]
    # Find the last valid j
    valid_mask = j_mask
    if tl.sum(valid_mask.to(tl.int32)) > 0:
        # Get the last valid value
        last_valid_idx = LEN_2D - 1
        if j_offset <= last_valid_idx < j_offset + BLOCK_SIZE:
            local_idx = last_valid_idx - j_offset
            a_final = a_new_vals[local_idx] if local_idx < BLOCK_SIZE and j_mask[local_idx] else a_new_vals[tl.sum(valid_mask.to(tl.int32)) - 1]
        else:
            a_final = a_new_vals[tl.sum(valid_mask.to(tl.int32)) - 1]
        
        # Store a[i] only once per kernel launch
        if pid == (LEN_2D - 1) // BLOCK_SIZE:
            tl.store(a_ptr + i_val, a_final)
    
    # Compute aa[j][i] = a[i] + bb[j][i]
    aa_new_vals = a_new_vals + bb_vals
    
    # Store aa[j][i] values
    tl.store(aa_ptrs, aa_new_vals, mask=j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential over i (1 to LEN_2D-1)
    for i in range(1, LEN_2D):
        # Process all j values in parallel for this i
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        
        # First pass: compute all a[i] values from different j iterations
        temp_a_vals = torch.zeros(LEN_2D, dtype=a.dtype, device=a.device)
        
        # Launch kernel to compute all aa[j][i] values and determine final a[i]
        for j_block in range(0, LEN_2D, BLOCK_SIZE):
            j_end = min(j_block + BLOCK_SIZE, LEN_2D)
            
            # Load a[i-1]
            a_prev = a[i-1]
            
            # Process this block of j values
            for j in range(j_block, j_end):
                # Compute a[i] = aa[j][i] - a[i-1]
                a_new = aa[j, i] - a_prev
                temp_a_vals[j] = a_new
                
                # Compute aa[j][i] = a[i] + bb[j][i]
                aa[j, i] = a_new + bb[j, i]
        
        # The final a[i] is from the last j iteration (j = LEN_2D - 1)
        a[i] = temp_a_vals[LEN_2D - 1]