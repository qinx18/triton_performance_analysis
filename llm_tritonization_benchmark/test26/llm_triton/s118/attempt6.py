import triton
import triton.language as tl

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i, len_2d, BLOCK_SIZE: tl.constexpr):
    # Calculate j indices for this i
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid j values (0 <= j <= i-1)
    j_mask = j_offsets < i
    
    # Load bb[j][i] values
    bb_ptrs = bb_ptr + j_offsets * len_2d + i
    bb_vals = tl.load(bb_ptrs, mask=j_mask, other=0.0)
    
    # Load a[i-j-1] values
    a_indices = i - j_offsets - 1
    a_ptrs = a_ptr + a_indices
    a_vals = tl.load(a_ptrs, mask=j_mask, other=0.0)
    
    # Compute products and sum
    products = bb_vals * a_vals
    result = tl.sum(products)
    
    # Load current a[i] and add the result
    current_a = tl.load(a_ptr + i)
    new_a = current_a + result
    tl.store(a_ptr + i, new_a)

def s118_triton(a, bb):
    len_2d = bb.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    
    for i in range(1, len_2d):
        s118_kernel[(1,)](a, bb, i, len_2d, BLOCK_SIZE=BLOCK_SIZE)