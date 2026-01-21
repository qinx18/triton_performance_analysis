import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, N, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < N
    
    # For each j, compute aa[j][i] = (aa[j][i] - a[i-1]) + bb[j][i]
    # Since all j iterations write to the same a[i], we compute the final result
    aa_col_ptrs = aa_ptr + j_offsets * N + i
    bb_col_ptrs = bb_ptr + j_offsets * N + i
    
    aa_vals = tl.load(aa_col_ptrs, mask=j_mask)
    bb_vals = tl.load(bb_col_ptrs, mask=j_mask)
    a_prev = tl.load(a_ptr + i - 1)
    
    # Compute a[i] values for each j
    a_vals = aa_vals - a_prev
    
    # Since all j iterations overwrite a[i], we need the last valid j's result
    # Use the maximum valid j index
    valid_j_mask = j_mask
    if tl.sum(valid_j_mask.to(tl.int32)) > 0:
        # Find the last valid j
        last_j = tl.max(tl.where(valid_j_mask, j_offsets, -1))
        final_a_val = tl.sum(tl.where(j_offsets == last_j, a_vals, 0.0))
        
        # Store final a[i] value
        tl.store(a_ptr + i, final_a_val)
        
        # Update aa[j][i] = a[i] + bb[j][i] for all j
        new_aa_vals = final_a_val + bb_vals
        tl.store(aa_col_ptrs, new_aa_vals, mask=j_mask)

def s257_triton(a, aa, bb):
    N = aa.shape[0]  # LEN_2D
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    # Sequential loop over i from 1 to N-1
    for i in range(1, N):
        # Create a kernel that processes all j values for current i
        @triton.jit
        def s257_kernel_i(a_ptr, aa_ptr, bb_ptr, i_val, N, BLOCK_SIZE: tl.constexpr):
            j_offsets = tl.arange(0, BLOCK_SIZE)
            j_mask = j_offsets < N
            
            # Load a[i-1]
            a_prev = tl.load(a_ptr + i_val - 1)
            
            # For each j, compute a[i] = aa[j][i] - a[i-1]
            aa_col_ptrs = aa_ptr + j_offsets * N + i_val
            bb_col_ptrs = bb_ptr + j_offsets * N + i_val
            
            aa_vals = tl.load(aa_col_ptrs, mask=j_mask)
            bb_vals = tl.load(bb_col_ptrs, mask=j_mask)
            
            # All j iterations overwrite a[i], so we take the last valid one
            a_vals = aa_vals - a_prev
            
            # Get the last valid j's result
            if tl.sum(j_mask.to(tl.int32)) > 0:
                # Find the maximum valid j index
                max_valid_j = tl.max(tl.where(j_mask, j_offsets, -1))
                
                # Get the a[i] value from the last j iteration
                final_a_val = tl.sum(tl.where(j_offsets == max_valid_j, a_vals, 0.0))
                
                # Store a[i] (only one thread should do this)
                if tl.program_id(0) == 0:
                    tl.store(a_ptr + i_val, final_a_val)
                
                # Update aa[j][i] = a[i] + bb[j][i] for all valid j
                new_aa_vals = final_a_val + bb_vals
                tl.store(aa_col_ptrs, new_aa_vals, mask=j_mask)
        
        # Launch kernel for current i
        s257_kernel_i[grid](a, aa, bb, i, N, BLOCK_SIZE=BLOCK_SIZE)