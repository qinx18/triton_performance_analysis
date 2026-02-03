import triton
import triton.language as tl

@triton.jit
def s118_kernel(a_ptr, bb_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(1, len_2d):
        j_mask = j_offsets < i
        
        # Load bb[j][i] for valid j values
        bb_ptrs = bb_ptr + j_offsets * len_2d + i
        bb_vals = tl.load(bb_ptrs, mask=j_mask, other=0.0)
        
        # Load a[i-j-1] for valid j values
        a_indices = i - j_offsets - 1
        a_ptrs = a_ptr + a_indices
        a_vals = tl.load(a_ptrs, mask=j_mask, other=0.0)
        
        # Compute products and sum
        products = bb_vals * a_vals
        sum_val = tl.sum(products)
        
        # Update a[i]
        a_i_ptr = a_ptr + i
        current_val = tl.load(a_i_ptr)
        tl.store(a_i_ptr, current_val + sum_val)

def s118_triton(a, bb, len_2d):
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    
    grid = (1,)
    
    s118_kernel[grid](
        a, bb, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )