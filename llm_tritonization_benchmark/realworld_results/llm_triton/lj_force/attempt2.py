import triton
import triton.language as tl
import torch

@triton.jit
def lj_force_kernel(
    f_ptr, neighbors_ptr, numneigh_ptr, pos_ptr,
    cutforcesq_val, epsilon_val, sigma6_val,
    MAXNEIGHS: tl.constexpr, NLOCAL: tl.constexpr, PAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID and compute atom indices for this block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    atom_ids = block_start + offsets
    mask = atom_ids < NLOCAL
    
    # Skip if no valid atoms in this block
    if tl.sum(mask) > 0:
        # Clear forces for atoms in this block
        for dim in range(PAD):
            force_offsets = atom_ids * PAD + dim
            tl.store(f_ptr + force_offsets, 0.0, mask=mask)
        
        # Process each valid atom in the block
        for ii in range(BLOCK_SIZE):
            i = block_start + ii
            if_valid = i < NLOCAL
            if if_valid:
                # Load atom i position
                xtmp = tl.load(pos_ptr + i * PAD + 0)
                ytmp = tl.load(pos_ptr + i * PAD + 1)
                ztmp = tl.load(pos_ptr + i * PAD + 2)
                
                # Initialize force accumulators
                fix = 0.0
                fiy = 0.0
                fiz = 0.0
                
                # Get number of neighbors for atom i
                num_neighbors = tl.load(numneigh_ptr + i)
                
                # Process neighbors of atom i
                for k in range(BLOCK_SIZE):
                    neighbor_valid = k < num_neighbors
                    if neighbor_valid:
                        # Get neighbor index
                        j = tl.load(neighbors_ptr + i * MAXNEIGHS + k)
                        
                        # Compute distance vector
                        delx = xtmp - tl.load(pos_ptr + j * PAD + 0)
                        dely = ytmp - tl.load(pos_ptr + j * PAD + 1)
                        delz = ztmp - tl.load(pos_ptr + j * PAD + 2)
                        rsq = delx * delx + dely * dely + delz * delz
                        
                        # Apply cutoff check
                        cutoff_valid = rsq < cutforcesq_val
                        if cutoff_valid:
                            sr2 = 1.0 / rsq
                            sr6 = sr2 * sr2 * sr2 * sigma6_val
                            force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon_val
                            fix += delx * force
                            fiy += dely * force
                            fiz += delz * force
                
                # Store accumulated forces
                tl.store(f_ptr + i * PAD + 0, fix)
                tl.store(f_ptr + i * PAD + 1, fiy)
                tl.store(f_ptr + i * PAD + 2, fiz)

def lj_force_triton(f, neighbors, numneigh, pos, cutforcesq_val, epsilon_val, sigma6_val, MAXNEIGHS, NLOCAL, PAD):
    BLOCK_SIZE = 256
    grid = (triton.cdiv(NLOCAL, BLOCK_SIZE),)
    
    lj_force_kernel[grid](
        f, neighbors, numneigh, pos,
        cutforcesq_val, epsilon_val, sigma6_val,
        MAXNEIGHS, NLOCAL, PAD, BLOCK_SIZE
    )