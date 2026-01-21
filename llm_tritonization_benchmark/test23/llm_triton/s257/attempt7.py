import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i, N, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < N
    
    # Load a[i-1] once
    a_i_prev = tl.load(a_ptr + (i - 1))
    
    # Load aa[j][i] values for all j
    aa_offsets = j_offsets * N + i
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask)
    bb_vals = tl.load(bb_ptr + aa_offsets, mask=mask)
    
    # For each j, a[i] = aa[j][i] - a[i-1]
    new_a_vals = aa_vals - a_i_prev
    
    # Since each j overwrites a[i], we need to process sequentially within the kernel
    # Process one j at a time to maintain correct overwrite semantics
    for block_j in range(N):
        if block_j < BLOCK_SIZE:
            j_mask = j_offsets == block_j
            if tl.any(j_mask):
                # Get the new a[i] value for this j
                a_val = tl.where(j_mask, new_a_vals, 0.0)
                a_val = tl.sum(a_val)
                
                # Store a[i]
                tl.store(a_ptr + i, a_val)
                
                # Update aa[j][i] = a[i] + bb[j][i]
                bb_val = tl.where(j_mask, bb_vals, 0.0)
                bb_val = tl.sum(bb_val)
                new_aa_val = a_val + bb_val
                
                aa_offset = block_j * N + i
                tl.store(aa_ptr + aa_offset, new_aa_val)

def s257_triton(a, aa, bb):
    N = aa.shape[0]  # LEN_2D
    BLOCK_SIZE = min(256, triton.next_power_of_2(N))
    
    # Sequential loop over i from 1 to N-1
    for i in range(1, N):
        # Process all j values, but need to handle overwrite correctly
        for j_start in range(0, N, BLOCK_SIZE):
            remaining = min(BLOCK_SIZE, N - j_start)
            
            # Load a[i-1]
            a_i_prev = a[i-1].item()
            
            # Process this block of j values
            for local_j in range(remaining):
                global_j = j_start + local_j
                # a[i] = aa[j][i] - a[i-1]
                a[i] = aa[global_j, i] - a_i_prev
                # aa[j][i] = a[i] + bb[j][i]
                aa[global_j, i] = a[i] + bb[global_j, i]
                # Update a_i_prev for potential next iteration
                a_i_prev = a[i].item()