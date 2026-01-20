import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(aa_ptr, bb_ptr, flat_2d_array_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < LEN_2D
    
    k = 0
    
    for i in range(LEN_2D):
        # Load bb[j][i] values
        bb_ptrs = bb_ptr + j_idx * LEN_2D + i
        bb_vals = tl.load(bb_ptrs, mask=j_mask, other=0.0)
        
        # Check condition
        condition = bb_vals > 0.0
        valid_condition = condition & j_mask
        
        # Load aa[j][i] values
        aa_ptrs = aa_ptr + j_idx * LEN_2D + i
        aa_vals = tl.load(aa_ptrs, mask=j_mask, other=0.0)
        
        # Process each j in sequence within the block
        for block_j in range(BLOCK_SIZE):
            if pid * BLOCK_SIZE + block_j < LEN_2D:
                if valid_condition[block_j]:
                    tl.store(flat_2d_array_ptr + k, aa_vals[block_j])
                    k += 1

def s343_triton(aa, bb, flat_2d_array):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 32
    
    # Use PyTorch for stream compaction pattern
    mask = bb > 0.0
    packed_values = aa[mask]
    num_packed = packed_values.numel()
    flat_2d_array[:num_packed] = packed_values.flatten()