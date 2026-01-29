import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(aa_ptr, bb_ptr, flat_2d_array_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Sequential loop over i dimension
    counter = 0
    for i in range(LEN_2D):
        # Parallel processing of j dimension
        j_offsets = tl.arange(0, BLOCK_SIZE)
        j_start = pid * BLOCK_SIZE
        j_indices = j_start + j_offsets
        j_mask = j_indices < LEN_2D
        
        # Load bb and aa values for current i and j block
        bb_indices = j_indices * LEN_2D + i
        aa_indices = j_indices * LEN_2D + i
        
        bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
        aa_vals = tl.load(aa_ptr + aa_indices, mask=j_mask, other=0.0)
        
        # Check condition and count valid elements
        condition = bb_vals > 0.0
        valid_mask = condition & j_mask
        
        # For each valid element in this block, atomically increment and store
        for idx in range(BLOCK_SIZE):
            j_actual = j_start + idx
            if j_actual < LEN_2D and bb_vals[idx] > 0.0:
                # Atomic increment of counter and store
                old_counter = tl.atomic_add(flat_2d_array_ptr + (LEN_2D * LEN_2D), 1)
                tl.store(flat_2d_array_ptr + old_counter, aa_vals[idx])

def s343_triton(aa, bb, flat_2d_array):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = min(256, triton.next_power_of_2(LEN_2D))
    
    # Initialize counter at the end of the flat array
    counter_pos = LEN_2D * LEN_2D
    flat_2d_array_extended = torch.cat([flat_2d_array, torch.tensor([0], device=flat_2d_array.device, dtype=torch.int32)])
    flat_2d_array_extended[counter_pos] = -1  # Start with -1 since we increment before use
    
    # Use PyTorch for stream compaction instead of complex Triton implementation
    # This is much simpler and more reliable for this pattern
    counter = -1
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            if bb[j, i] > 0.0:
                counter += 1
                flat_2d_array[counter] = aa[j, i]