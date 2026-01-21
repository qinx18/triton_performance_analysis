import triton
import triton.language as tl

@triton.jit
def s126_kernel(bb_ptr, cc_ptr, flat_2d_array_ptr, N: tl.constexpr):
    # Process one column (i) per program
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    # Start with k = 1 + i * (N - 1) + i (since k increments N times per i iteration)
    k = 1 + i * N
    
    # Sequential loop over j from 1 to N-1
    for j in range(1, N):
        # Load bb[j-1][i]
        prev_bb_offset = (j - 1) * N + i
        prev_bb = tl.load(bb_ptr + prev_bb_offset)
        
        # Load flat_2d_array[k-1]
        flat_val = tl.load(flat_2d_array_ptr + (k - 1))
        
        # Load cc[j][i]
        cc_offset = j * N + i
        cc_val = tl.load(cc_ptr + cc_offset)
        
        # Compute bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i]
        result = prev_bb + flat_val * cc_val
        
        # Store bb[j][i]
        bb_offset = j * N + i
        tl.store(bb_ptr + bb_offset, result)
        
        # Increment k (once per j iteration)
        k += 1
    
    # Additional increment of k after finishing the j loop (as in C code)
    # This doesn't affect computation but matches the C indexing pattern

def s126_triton(bb, cc, flat_2d_array):
    N = bb.shape[0]
    
    # Launch kernel with one program per column
    grid = (N,)
    s126_kernel[grid](bb, cc, flat_2d_array, N=N)