import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, a_copy_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Single thread processes all i values sequentially
    if tl.program_id(0) != 0:
        return
    
    # Process each i value sequentially due to RAW dependency
    for i in range(1, LEN_2D):
        # Load a[i-1] from the read-only copy
        a_prev = tl.load(a_copy_ptr + (i - 1))
        
        # Process all j values in parallel using vectorized operations
        j_offsets = tl.arange(0, BLOCK_SIZE)
        
        # Process j values in blocks
        for j_start in range(0, LEN_2D, BLOCK_SIZE):
            current_j = j_start + j_offsets
            j_mask = current_j < LEN_2D
            
            # Load aa[j][i] and bb[j][i] for current block of j values
            aa_ptrs = aa_ptr + current_j * LEN_2D + i
            bb_ptrs = bb_ptr + current_j * LEN_2D + i
            
            aa_vals = tl.load(aa_ptrs, mask=j_mask)
            bb_vals = tl.load(bb_ptrs, mask=j_mask)
            
            # Compute a[i] = aa[j][i] - a[i-1] (same for all j)
            a_val = aa_vals - a_prev
            
            # Compute aa[j][i] = a[i] + bb[j][i]
            new_aa_vals = a_val + bb_vals
            
            # Store results
            tl.store(aa_ptrs, new_aa_vals, mask=j_mask)
        
        # Store the final a[i] value (use the last computed a_val)
        # Since a[i] is the same for all j, we can use any of the computed values
        j_offsets_single = tl.arange(0, 1)
        aa_ptr_single = aa_ptr + 0 * LEN_2D + i
        aa_val_single = tl.load(aa_ptr_single)
        bb_ptr_single = bb_ptr + 0 * LEN_2D + i  
        bb_val_single = tl.load(bb_ptr_single)
        final_a_val = aa_val_single - a_prev
        
        tl.store(a_ptr + i, final_a_val)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    # Launch kernel with single thread to handle sequential dependency
    grid = (1,)
    s257_kernel[grid](
        a, aa, bb, a_copy,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )