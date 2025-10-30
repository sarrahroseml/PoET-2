from __future__ import annotations
import os 
from pathlib import Path
from typing import List, Tuple
from Bio.PDB.Polypeptide import three_to_index, index_to_one
import numpy as np
import gemmi

IS_ALPHAFOLD = True  # set True when pLDDT scores stored in B-factor
INPUT = "/n/groups/marks/projects/viral_plm/models/PoET-2/data/structures/pdb_files/viralaf2"
OUT = "/n/groups/marks/projects/viral_plm/models/PoET-2/data/structures/structure_arrays"
def residue_backbone(residue: gemmi.Residue) -> Tuple[np.ndarray, float]: 
    coords = np.full((3,3), np.nan, dtype=np.float32)
    score = np.nan
    try: 
        aa = three_to_index(residue.name)
        aa = index_to_one(aa)
    except KeyError: 
        aa = "X"

    atoms = {atom.name.strip(): atom for atom in residue}
    backbone = ["N", "CA", "C"]
    missing_backbone = False 
    for idx, atom_name in enumerate(backbone): 
        atom = atoms.get(atom_name)
        if atom is None: 
            missing_backbone = True
            continue 
        pos = atom.pos
        coords[idx] = (pos.x, pos.y, pos.z)
    if missing_backbone: 
        return coords, score, aa
    if IS_ALPHAFOLD and "CA" in atoms: 
        score = atoms["CA"].b_iso
    
    return coords, score, aa

def build_arrays(path: Path) -> Tuple[np.ndarray, np.ndarray]: 
    structure = gemmi.read_structure(str(path))
    chain = structure[0][0]
    length = (len(chain))
    seq_chars = []
    coords = np.full((length, 3, 3), np.nan, dtype = np.float32)
    scores = np.full((length, ), np.nan, dtype = np.float32)
    for i, res in enumerate(chain): 
        coord, score, aa = residue_backbone(res)
        coords[i] = coord
        scores[i] = score
        seq_chars.append(aa)
    seq = "".join(seq_chars)
    return coords, scores, seq

def main(): 
    for filename in os.listdir(INPUT):
        name = filename.split(".")[0]
        full_path = os.path.join(INPUT, filename)
        if os.path.isfile(full_path):
            print(f"File: {filename}")
            coords, scores, seq = build_arrays(full_path)
            sequence=np.array(seq, dtype="U"),
            np.savez_compressed(f"{OUT}/{name}.npz", coords=coords, scores=scores, sequence=sequence)

    


if __name__ == "__main__":
     main()
    
    

