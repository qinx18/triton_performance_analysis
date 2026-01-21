import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate j range for this i
    j_start = pid * BLOCK_SIZE
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < i_val
    
    # Load bb[j][i] values - bb is stored in row-major order
    bb_indices = j_offsets * N + i_val
    bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
    
    # Load a[i-j-1] values
    a_read_indices = i_val - j_offsets - 1
    a_read_mask = j_mask & (a_read_indices >= 0)
    a_vals = tl.load(a_ptr + a_read_indices, mask=a_read_mask, other=0.0)
    
    # Compute products and sum
    products = tl.where(a_read_mask, bb_vals * a_vals, 0.0)
    result = tl.sum(products)
    
    # Store result using atomic add
    if result != 0.0:
        if pid == 0:
            # Use separate kernel to avoid atomic contention
            pass

@triton.jit
def s118_add_kernel(a_ptr, result, i_val):
    tl.store(a_ptr + i_val, tl.load(a_ptr + i_val) + result)

@triton.jit
def s118_sum_kernel(a_ptr, bb_ptr, i_val, N, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    total_sum = tl.zeros([1], dtype=tl.float32)
    
    # Process all j values for this i in chunks
    for j_start in range(0, i_val, BLOCK_SIZE):
        current_j = j_start + j_offsets
        j_mask = current_j < i_val
        
        # Load bb[j][i] values
        bb_indices = current_j * N + i_val
        bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
        
        # Load a[i-j-1] values
        a_read_indices = i_val - current_j - 1
        a_read_mask = j_mask & (a_read_indices >= 0)
        a_vals = tl.load(a_ptr + a_read_indices, mask=a_read_mask, other=0.0)
        
        # Compute products and accumulate
        products = tl.where(a_read_mask, bb_vals * a_vals, 0.0)
        total_sum += tl.sum(products)
    
    # Add to a[i]
    current_a = tl.load(a_ptr + i_val)
    tl.store(a_ptr + i_val, current_a + total_sum)

def s118_triton(a, bb):
    N = bb.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over i
    for i in range(1, N):
        grid = (1,)
        s118_sum_kernel[grid](a, bb, i, N, BLOCK_SIZE)
    
    return a