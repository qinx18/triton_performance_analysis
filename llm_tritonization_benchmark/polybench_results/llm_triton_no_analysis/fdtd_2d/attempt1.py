import triton
import triton.language as tl
import torch

@triton.jit
def fdtd_2d_kernel(
    _fict_ptr, ex_ptr, ey_ptr, hz_ptr,
    NX: tl.constexpr, NY: tl.constexpr, TMAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    for t in range(TMAX):
        # Step 1: ey[0][j] = _fict_[t] for j = 0 to NY-1
        block_start = tl.program_id(0) * BLOCK_SIZE
        offsets = tl.arange(0, BLOCK_SIZE)
        j_offsets = block_start + offsets
        mask = j_offsets < NY
        
        fict_val = tl.load(_fict_ptr + t)
        ey_offsets = 0 * NY + j_offsets  # i=0, j varies
        tl.store(ey_ptr + ey_offsets, fict_val, mask=mask)
        
        # Step 2: ey[i][j] = ey[i][j] - 0.5*(hz[i][j]-hz[i-1][j]) for i=1 to NX-1, j=0 to NY-1
        pid_x = tl.program_id(0)
        pid_y = tl.program_id(1)
        
        i_start = 1 + pid_x * BLOCK_SIZE
        j_start = pid_y * BLOCK_SIZE
        
        i_offsets = tl.arange(0, BLOCK_SIZE)
        j_offsets = tl.arange(0, BLOCK_SIZE)
        
        i_indices = i_start + i_offsets[:, None]
        j_indices = j_start + j_offsets[None, :]
        
        mask_ey = (i_indices < NX) & (j_indices < NY)
        
        ey_idx = i_indices * NY + j_indices
        hz_idx = i_indices * NY + j_indices
        hz_prev_idx = (i_indices - 1) * NY + j_indices
        
        ey_vals = tl.load(ey_ptr + ey_idx, mask=mask_ey)
        hz_vals = tl.load(hz_ptr + hz_idx, mask=mask_ey)
        hz_prev_vals = tl.load(hz_ptr + hz_prev_idx, mask=mask_ey)
        
        new_ey = ey_vals - 0.5 * (hz_vals - hz_prev_vals)
        tl.store(ey_ptr + ey_idx, new_ey, mask=mask_ey)
        
        # Step 3: ex[i][j] = ex[i][j] - 0.5*(hz[i][j]-hz[i][j-1]) for i=0 to NX-1, j=1 to NY-1
        i_start_ex = pid_x * BLOCK_SIZE
        j_start_ex = 1 + pid_y * BLOCK_SIZE
        
        i_indices_ex = i_start_ex + i_offsets[:, None]
        j_indices_ex = j_start_ex + j_offsets[None, :]
        
        mask_ex = (i_indices_ex < NX) & (j_indices_ex < NY)
        
        ex_idx = i_indices_ex * NY + j_indices_ex
        hz_idx_ex = i_indices_ex * NY + j_indices_ex
        hz_prev_j_idx = i_indices_ex * NY + (j_indices_ex - 1)
        
        ex_vals = tl.load(ex_ptr + ex_idx, mask=mask_ex)
        hz_vals_ex = tl.load(hz_ptr + hz_idx_ex, mask=mask_ex)
        hz_prev_j_vals = tl.load(hz_ptr + hz_prev_j_idx, mask=mask_ex)
        
        new_ex = ex_vals - 0.5 * (hz_vals_ex - hz_prev_j_vals)
        tl.store(ex_ptr + ex_idx, new_ex, mask=mask_ex)
        
        # Step 4: hz[i][j] = hz[i][j] - 0.7*(ex[i][j+1] - ex[i][j] + ey[i+1][j] - ey[i][j])
        # for i=0 to NX-2, j=0 to NY-2
        i_start_hz = pid_x * BLOCK_SIZE
        j_start_hz = pid_y * BLOCK_SIZE
        
        i_indices_hz = i_start_hz + i_offsets[:, None]
        j_indices_hz = j_start_hz + j_offsets[None, :]
        
        mask_hz = (i_indices_hz < (NX - 1)) & (j_indices_hz < (NY - 1))
        
        hz_idx_update = i_indices_hz * NY + j_indices_hz
        ex_idx_curr = i_indices_hz * NY + j_indices_hz
        ex_idx_next_j = i_indices_hz * NY + (j_indices_hz + 1)
        ey_idx_curr = i_indices_hz * NY + j_indices_hz
        ey_idx_next_i = (i_indices_hz + 1) * NY + j_indices_hz
        
        hz_vals_update = tl.load(hz_ptr + hz_idx_update, mask=mask_hz)
        ex_curr = tl.load(ex_ptr + ex_idx_curr, mask=mask_hz)
        ex_next_j = tl.load(ex_ptr + ex_idx_next_j, mask=mask_hz)
        ey_curr = tl.load(ey_ptr + ey_idx_curr, mask=mask_hz)
        ey_next_i = tl.load(ey_ptr + ey_idx_next_i, mask=mask_hz)
        
        new_hz = hz_vals_update - 0.7 * (ex_next_j - ex_curr + ey_next_i - ey_curr)
        tl.store(hz_ptr + hz_idx_update, new_hz, mask=mask_hz)

def fdtd_2d_triton(_fict_, ex, ey, hz, NX, NY, TMAX):
    BLOCK_SIZE = 16
    
    # For step 1 (boundary condition), we need only 1D grid for j dimension
    grid_1d = (triton.cdiv(NY, BLOCK_SIZE),)
    
    # For steps 2, 3, 4, we need 2D grid
    grid_2d = (triton.cdiv(NX, BLOCK_SIZE), triton.cdiv(NY, BLOCK_SIZE))
    
    # Launch with the largest grid needed
    fdtd_2d_kernel[grid_2d](
        _fict_, ex, ey, hz,
        NX, NY, TMAX, BLOCK_SIZE
    )