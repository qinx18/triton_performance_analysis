import triton
import triton.language as tl
import torch

@triton.jit
def s118_kernel(a_ptr, bb_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(1, len_2d):
        mask = j_offsets <= i - 1
        
        # Load bb[j][i] values
        bb_indices = j_offsets * len_2d + i
        bb_vals = tl.load(bb_ptr + bb_indices, mask=mask, other=0.0)
        
        # Load a[i-j-1] values
        a_read_indices = i - j_offsets - 1
        a_vals = tl.load(a_ptr + a_read_indices, mask=mask, other=0.0)
        
        # Compute products
        products = bb_vals * a_vals
        
        # Sum across j dimension
        result = tl.sum(products)
        
        # Load current a[i] and add result
        current_a = tl.load(a_ptr + i)
        new_a = current_a + result
        
        # Store back to a[i]
        tl.store(a_ptr + i, new_a)

def s118_triton(a, bb, len_2d):
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    
    grid = (1,)
    s118_kernel[grid](a, bb, len_2d, BLOCK_SIZE)