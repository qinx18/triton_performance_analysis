import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i, N, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < N
    
    # Load a[i-1] once
    a_i_prev = tl.load(a_ptr + (i - 1))
    
    # We need to process each j sequentially because a[i] gets updated in each iteration
    a_i = a_i_prev
    for j in range(N):
        # a[i] = aa[j][i] - a[i-1]
        aa_ji_offset = j * N + i
        aa_ji_val = tl.load(aa_ptr + aa_ji_offset)
        a_i = aa_ji_val - a_i_prev
        
        # Store the updated a[i]
        tl.store(a_ptr + i, a_i)
        
        # aa[j][i] = a[i] + bb[j][i]
        bb_ji_val = tl.load(bb_ptr + aa_ji_offset)
        new_aa_ji = a_i + bb_ji_val
        tl.store(aa_ptr + aa_ji_offset, new_aa_ji)

def s257_triton(a, aa, bb):
    N = aa.shape[0]  # LEN_2D
    BLOCK_SIZE = triton.next_power_of_2(N)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    
    # Sequential loop over i from 1 to N-1
    for i in range(1, N):
        grid = (1,)
        s257_kernel[grid](a, aa, bb, i, N, BLOCK_SIZE=BLOCK_SIZE)