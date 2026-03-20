import triton
import triton.language as tl
import torch

@triton.jit
def fdtd_2d_kernel(
    _fict_ptr, ex_ptr, ey_ptr, hz_ptr,
    NX: tl.constexpr, NY: tl.constexpr, TMAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get program IDs
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    # Pre-compute offsets once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for t in range(TMAX):
        # Step 1: ey[0][j] = _fict_[t] for j = 0 to NY-1
        if pid_x == 0:  # Only first block in x dimension handles this
            j_start = pid_y * BLOCK_SIZE
            j_offsets = j_start + offsets
            j_mask = j_offsets < NY
            
            fict_val = tl.load(_fict_ptr + t)
            ey_idx = 0 * NY + j_offsets  # i=0, j varies
            tl.store(ey_ptr + ey_idx, fict_val, mask=j_mask)
        
        # Synchronization barrier
        tl.debug_barrier()
        
        # Step 2: ey[i][j] = ey[i][j] - 0.5*(hz[i][j]-hz[i-1][j]) for i=1 to NX-1, j=0 to NY-1
        i_start = 1 + pid_x * BLOCK_SIZE
        j_start = pid_y * BLOCK_SIZE
        
        i_offsets = i_start + offsets[:, None]
        j_offsets = j_start + offsets[None, :]
        
        i_mask = i_offsets < NX
        j_mask = j_offsets < NY
        mask_ey = i_mask & j_mask
        
        ey_idx = i_offsets * NY + j_offsets
        hz_idx = i_offsets * NY + j_offsets
        hz_prev_idx = (i_offsets - 1) * NY + j_offsets
        
        ey_vals = tl.load(ey_ptr + ey_idx, mask=mask_ey)
        hz_vals = tl.load(hz_ptr + hz_idx, mask=mask_ey)
        hz_prev_vals = tl.load(hz_ptr + hz_prev_idx, mask=mask_ey)
        
        new_ey = ey_vals - 0.5 * (hz_vals - hz_prev_vals)
        tl.store(ey_ptr + ey_idx, new_ey, mask=mask_ey)
        
        tl.debug_barrier()
        
        # Step 3: ex[i][j] = ex[i][j] - 0.5*(hz[i][j]-hz[i][j-1]) for i=0 to NX-1, j=1 to NY-1
        i_start_ex = pid_x * BLOCK_SIZE
        j_start_ex = 1 + pid_y * BLOCK_SIZE
        
        i_offsets_ex = i_start_ex + offsets[:, None]
        j_offsets_ex = j_start_ex + offsets[None, :]
        
        i_mask_ex = i_offsets_ex < NX
        j_mask_ex = j_offsets_ex < NY
        mask_ex = i_mask_ex & j_mask_ex
        
        ex_idx = i_offsets_ex * NY + j_offsets_ex
        hz_idx_ex = i_offsets_ex * NY + j_offsets_ex
        hz_prev_j_idx = i_offsets_ex * NY + (j_offsets_ex - 1)
        
        ex_vals = tl.load(ex_ptr + ex_idx, mask=mask_ex)
        hz_vals_ex = tl.load(hz_ptr + hz_idx_ex, mask=mask_ex)
        hz_prev_j_vals = tl.load(hz_ptr + hz_prev_j_idx, mask=mask_ex)
        
        new_ex = ex_vals - 0.5 * (hz_vals_ex - hz_prev_j_vals)
        tl.store(ex_ptr + ex_idx, new_ex, mask=mask_ex)
        
        tl.debug_barrier()
        
        # Step 4: hz[i][j] = hz[i][j] - 0.7*(ex[i][j+1] - ex[i][j] + ey[i+1][j] - ey[i][j])
        # for i=0 to NX-2, j=0 to NY-2
        i_start_hz = pid_x * BLOCK_SIZE
        j_start_hz = pid_y * BLOCK_SIZE
        
        i_offsets_hz = i_start_hz + offsets[:, None]
        j_offsets_hz = j_start_hz + offsets[None, :]
        
        i_mask_hz = i_offsets_hz < (NX - 1)
        j_mask_hz = j_offsets_hz < (NY - 1)
        mask_hz = i_mask_hz & j_mask_hz
        
        hz_idx_update = i_offsets_hz * NY + j_offsets_hz
        ex_idx_curr = i_offsets_hz * NY + j_offsets_hz
        ex_idx_next_j = i_offsets_hz * NY + (j_offsets_hz + 1)
        ey_idx_curr = i_offsets_hz * NY + j_offsets_hz
        ey_idx_next_i = (i_offsets_hz + 1) * NY + j_offsets_hz
        
        hz_vals_update = tl.load(hz_ptr + hz_idx_update, mask=mask_hz)
        ex_curr = tl.load(ex_ptr + ex_idx_curr, mask=mask_hz)
        ex_next_j = tl.load(ex_ptr + ex_idx_next_j, mask=mask_hz)
        ey_curr = tl.load(ey_ptr + ey_idx_curr, mask=mask_hz)
        ey_next_i = tl.load(ey_ptr + ey_idx_next_i, mask=mask_hz)
        
        new_hz = hz_vals_update - 0.7 * (ex_next_j - ex_curr + ey_next_i - ey_curr)
        tl.store(hz_ptr + hz_idx_update, new_hz, mask=mask_hz)

def fdtd_2d_triton(_fict_, ex, ey, hz, NX, NY, TMAX):
    BLOCK_SIZE = 16
    
    grid = (triton.cdiv(NX, BLOCK_SIZE), triton.cdiv(NY, BLOCK_SIZE))
    
    fdtd_2d_kernel[grid](
        _fict_, ex, ey, hz,
        NX, NY, TMAX, BLOCK_SIZE
    )