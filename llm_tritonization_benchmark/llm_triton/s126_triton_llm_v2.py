import torch
import triton
import triton.language as tl

@triton.jit
def s126_kernel(
    bb_ptr, cc_ptr, flat_2d_array_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each block handles a contiguous chunk of i dimension
    block_id = tl.program_id(0)
    i_base = block_id * BLOCK_SIZE
    i_offsets = i_base + tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < LEN_2D
    
    # In-kernel sequential loop over j
    for j in range(1, LEN_2D):
        # Calculate k index: k = i * LEN_2D + j
        k_offsets = i_offsets * LEN_2D + j
        
        # Current and previous row offsets
        curr_offsets = j * LEN_2D + i_offsets
        prev_offsets = (j - 1) * LEN_2D + i_offsets
        
        # Load values
        bb_prev = tl.load(bb_ptr + prev_offsets, mask=i_mask)
        cc_curr = tl.load(cc_ptr + curr_offsets, mask=i_mask)
        flat_val = tl.load(flat_2d_array_ptr + k_offsets - 1, mask=i_mask)
        
        # Compute result
        result = bb_prev + flat_val * cc_curr
        
        # Store result
        tl.store(bb_ptr + curr_offsets, result, mask=i_mask)

def s126_triton(bb, cc, flat_2d_array):
    bb = bb.contiguous()
    cc = cc.contiguous()
    flat_2d_array = flat_2d_array.contiguous()
    
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s126_kernel[grid](
        bb, cc, flat_2d_array,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return bb