import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, j_offset, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = j_offset + tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Load aa[j][i] and bb[j][i] for current i
    aa_idx = j_offsets * LEN_2D + tl.load(tl.from_ptr([0], dtype=tl.int32, shape=[1]))  # i will be passed as offset
    bb_idx = j_offsets * LEN_2D + tl.load(tl.from_ptr([0], dtype=tl.int32, shape=[1]))  # i will be passed as offset
    
    aa_vals = tl.load(aa_ptr + aa_idx, mask=j_mask, other=0.0)
    bb_vals = tl.load(bb_ptr + bb_idx, mask=j_mask, other=0.0)
    
    # Load a[i-1]
    a_prev = tl.load(tl.from_ptr([0], dtype=tl.float32, shape=[1]))  # a[i-1] will be passed
    
    # Compute a[i] = aa[j][i] - a[i-1] (overwrite pattern - last j wins)
    a_new = aa_vals - a_prev
    
    # Compute aa[j][i] = a[i] + bb[j][i] 
    aa_new = a_new + bb_vals
    
    # Store aa[j][i]
    tl.store(aa_ptr + aa_idx, aa_new, mask=j_mask)

@triton.jit  
def s257_kernel_simple(a_ptr, aa_ptr, bb_ptr, i, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Calculate 2D indices for aa[j][i] and bb[j][i]
    aa_indices = j_offsets * LEN_2D + i
    bb_indices = j_offsets * LEN_2D + i
    
    # Load values
    aa_vals = tl.load(aa_ptr + aa_indices, mask=j_mask, other=0.0)
    bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
    a_prev = tl.load(a_ptr + (i - 1))
    
    # Compute a[i] = aa[j][i] - a[i-1] for each j (last j will overwrite)
    a_new_vals = aa_vals - a_prev
    
    # Compute aa[j][i] = a[i] + bb[j][i]
    aa_new_vals = a_new_vals + bb_vals
    
    # Store results
    tl.store(aa_ptr + aa_indices, aa_new_vals, mask=j_mask)
    
    # Store final a[i] value (from last valid j)
    if LEN_2D > 0:
        final_a = tl.load(aa_ptr + (LEN_2D - 1) * LEN_2D + i) - tl.load(bb_ptr + (LEN_2D - 1) * LEN_2D + i)
        tl.store(a_ptr + i, final_a)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over i, parallel over j
    for i in range(1, LEN_2D):
        # Load a[i-1]
        a_prev = a[i-1].item()
        
        # Process all j values in parallel
        for j_start in range(0, LEN_2D, BLOCK_SIZE):
            j_end = min(j_start + BLOCK_SIZE, LEN_2D)
            actual_block_size = j_end - j_start
            
            if actual_block_size <= BLOCK_SIZE:
                grid = (1,)
                s257_kernel_simple[grid](
                    a, aa, bb, i, LEN_2D, BLOCK_SIZE=BLOCK_SIZE
                )
                break
        
        # Update a[i] with the result from the last j iteration
        # Since all j iterations write to a[i], we need the final value
        # which comes from the computation: aa[j][i] - a[i-1]  
        # But aa[j][i] gets updated to a[i] + bb[j][i]
        # So a[i] should be aa[last_j][i] - bb[last_j][i]
        last_j = LEN_2D - 1
        a[i] = aa[last_j, i] - bb[last_j, i]