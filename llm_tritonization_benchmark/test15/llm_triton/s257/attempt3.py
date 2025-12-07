import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, a_copy_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Single thread processes all i values sequentially
    if tl.program_id(0) != 0:
        return
    
    # Pre-compute offsets once
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process each i value sequentially due to RAW dependency
    for i in range(1, LEN_2D):
        # Load a[i-1] from the read-only copy
        a_prev = tl.load(a_copy_ptr + (i - 1))
        
        # Initialize a[i] with first j=0 computation
        aa_0_ptr = aa_ptr + i  # aa[0][i]
        aa_0_val = tl.load(aa_0_ptr)
        a_new = aa_0_val - a_prev
        
        # Store a[i]
        tl.store(a_ptr + i, a_new)
        
        # Update aa[0][i] first
        bb_0_ptr = bb_ptr + i  # bb[0][i] 
        bb_0_val = tl.load(bb_0_ptr)
        tl.store(aa_0_ptr, a_new + bb_0_val)
        
        # Process remaining j values (j=1 to LEN_2D-1) in blocks
        for j_start in range(1, LEN_2D, BLOCK_SIZE):
            current_j = j_start + j_offsets
            j_mask = current_j < LEN_2D
            
            # Load aa[j][i] and bb[j][i] for current block of j values
            aa_ptrs = aa_ptr + current_j * LEN_2D + i
            bb_ptrs = bb_ptr + current_j * LEN_2D + i
            
            aa_vals = tl.load(aa_ptrs, mask=j_mask)
            bb_vals = tl.load(bb_ptrs, mask=j_mask)
            
            # Compute new a[i] using first valid j in this block
            if j_start == 1:
                # For j=1, update a[i] again
                first_aa = tl.load(aa_ptr + LEN_2D + i)  # aa[1][i]
                a_new = first_aa - a_prev
                tl.store(a_ptr + i, a_new)
            
            # Compute aa[j][i] = a[i] + bb[j][i]
            new_aa_vals = a_new + bb_vals
            
            # Store results
            tl.store(aa_ptrs, new_aa_vals, mask=j_mask)

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