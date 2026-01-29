import triton
import triton.language as tl
import torch

@triton.jit
def s126_kernel(bb_ptr, cc_ptr, flat_2d_array_ptr, N):
    # Process one column (i) per thread block
    i = tl.program_id(0)
    
    if i >= N:
        return
        
    # Sequential loop over j dimension due to recurrence dependency
    for j in range(1, N):
        # Calculate k index: k starts at 1, increments by 1 for each j iteration,
        # and has an extra increment for each i iteration
        k = 1 + j - 1 + i * (N + 1)
        
        # Load bb[j-1][i] (previous row, same column)
        bb_prev_offset = (j - 1) * N + i
        bb_prev = tl.load(bb_ptr + bb_prev_offset)
        
        # Load flat_2d_array[k-1]
        flat_offset = k - 1
        flat_val = tl.load(flat_2d_array_ptr + flat_offset)
        
        # Load cc[j][i]
        cc_offset = j * N + i
        cc_val = tl.load(cc_ptr + cc_offset)
        
        # Compute and store bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i]
        bb_offset = j * N + i
        result = bb_prev + flat_val * cc_val
        tl.store(bb_ptr + bb_offset, result)

def s126_triton(bb, cc, flat_2d_array):
    N = bb.shape[0]
    
    # Launch one thread block per column (i dimension)
    grid = (N,)
    
    s126_kernel[grid](
        bb, cc, flat_2d_array, N
    )