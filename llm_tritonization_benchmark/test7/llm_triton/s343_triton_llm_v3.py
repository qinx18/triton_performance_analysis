import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(aa_ptr, bb_ptr, flat_2d_array_ptr, LEN_2D):
    # Get program ID
    pid = tl.program_id(0)
    
    # Each program handles one output index
    if pid >= LEN_2D * LEN_2D:
        return
    
    # Calculate i and j from linear index
    i = pid // LEN_2D
    j = pid % LEN_2D
    
    # Count elements before this position that satisfy condition
    count = 0
    for ii in range(LEN_2D):
        for jj in range(LEN_2D):
            # Check if bb[jj][ii] > 0
            bb_idx = jj * LEN_2D + ii
            bb_val = tl.load(bb_ptr + bb_idx)
            
            if bb_val > 0.0:
                if ii < i or (ii == i and jj < j):
                    count += 1
                elif ii == i and jj == j:
                    # This is our position, store the value
                    aa_idx = jj * LEN_2D + ii
                    aa_val = tl.load(aa_ptr + aa_idx)
                    tl.store(flat_2d_array_ptr + count, aa_val)
                    return

def s343_triton(aa, bb, flat_2d_array):
    LEN_2D = aa.shape[0]
    
    # Initialize flat array to zero
    flat_2d_array.zero_()
    
    # First pass: count total valid elements
    valid_mask = bb > 0.0
    total_valid = valid_mask.sum().item()
    
    if total_valid == 0:
        return
    
    # Get indices of valid elements in column-major order
    valid_indices = []
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            if bb[j, i] > 0.0:
                valid_indices.append((i, j))
    
    # Pack the values
    for k, (i, j) in enumerate(valid_indices):
        flat_2d_array[k] = aa[j, i]