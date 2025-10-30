import numpy as np 


data = np.load("/n/groups/marks/projects/viral_plm/models/PoET-2/data/structures/structure_arrays/3JCL.npz")
coords = data["coords"]  # always present

print(coords.shape)

import gemmi

path = "/n/groups/marks/projects/viral_plm/models/PoET-2/data/structures/pdb_files/pdb/3JCL_B.pdb"  # download first: curl -O https://files.rcsb.org/view/3JCL.pdb
structure = gemmi.read_structure(path)

count = 0
for model in structure:
    for chain in model:
        for residue in chain:
            atoms = {atom.name.strip() for atom in residue}
            if {"N", "CA", "C"}.issubset(atoms):
                count += 1

print("Residues with full backbone:", count)
