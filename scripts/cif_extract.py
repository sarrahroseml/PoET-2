import numpy as np, gemmi
from pathlib import Path
from collections import defaultdict, Counter

# how to treat modified residues
MOD_MAP = {"MSE":"M", "SEC":"U", "PYL":"O"} 
OUT = "/n/groups/marks/projects/viral_plm/models/PoET-2/data/structures/structure_arrays"
INPUT = "/n/groups/marks/projects/viral_plm/models/PoET-2/data/structures/pdb_files/pdb/"
def one_letter(mon_id: str) -> str:
    if mon_id in MOD_MAP:
        return MOD_MAP[mon_id]
    cc = gemmi.find_tabulated_residue(mon_id)
    return cc.one_letter_code if (cc and cc.is_amino_acid()) else "X"

def auth_to_label_map(doc):
    # rows: [label_asym_id, auth_asym_id]
    # [['AA', 'Y'], ['AA', 'Y'], ['AB', 'Y'], ...]
    rows = doc.find(["_atom_site.label_asym_id","_atom_site.auth_asym_id"])
    buckets = defaultdict(Counter)
    for lab, auth in rows:
        buckets[auth].update([lab])
    # pick the most common label per author chain
    return {auth: cnt.most_common(1)[0][0] for auth, cnt in buckets.items()}

def seqres_aligned_coords(cif_path: Path, chain_id="A"):
    """
    Aligns 3D coordinates from ATOM records to full SEQRES sequence
    """
    # read SEQRES and mapping table
    doc = gemmi.cif.read(str(cif_path)).sole_block()
    print(doc)
    scheme = doc.find(
        ["_pdbx_poly_seq_scheme.asym_id",
         "_pdbx_poly_seq_scheme.mon_id",
         "_pdbx_poly_seq_scheme.seq_id",        # SEQRES index (label_seq_id)
         "_pdbx_poly_seq_scheme.auth_seq_num",  # author residue number
         "_pdbx_poly_seq_scheme.pdb_ins_code"]  # insertion code or '?'
    )

    scheme_ids = {r[0].strip() for r in scheme}
    cid = chain_id.strip()
    if cid not in scheme_ids: 
        amap = auth_to_label_map(doc)
        if cid in amap and amap[cid] in scheme_ids:
            cid = amap[cid]
        else:
            raise ValueError(f"Chain {cid} not found. Try one of: {sorted(scheme_ids)}")
    rows = [r for r in scheme if r[0].strip() == cid]

    if not rows:
        print(f"My chain id is {chain_id}")
        raise ValueError(f"Chain {chain_id} not found in mapping table")

    # intialise length, sequence and coords 
    L = max(int(r[2]) for r in rows)  # length from SEQRES index
    seq = ["?"] * L
    coords = np.full((L, 3, 3), np.nan, dtype=np.float32)
    scores = np.full((L,), np.nan, dtype=np.float32)

    # index ATOM residues by (auth_seq_num, icode)
    structure = gemmi.read_structure(str(cif_path))
    
    chain = structure[0][chain_id]
    idx = {}
    for res in chain:
        if res.is_water():
            continue
        if res.het_flag != ' ' and res.name not in MOD_MAP:
            continue
        key = (str(res.seqid.num), res.seqid.icode if res.seqid.icode else "?")
        idx[key] = res

    # iterates through each seq res entry 
    for asym_id, mon_id, seq_id, auth_num, icode in rows:
        i = int(seq_id) - 1
        # assign its amino acid
        seq[i] = one_letter(mon_id)
        # fill coordinates if residue present
        res = idx.get((auth_num, icode if icode != "?" else "?"))
        if res:
            atoms = {a.name.strip(): a for a in res}
            for k, name in enumerate(("N", "CA", "C")):
                a = atoms.get(name)
                if a:
                    p = a.pos
                    coords[i, k] = (p.x, p.y, p.z)

    return "".join(seq), coords, scores


def process_and_save(name: Path, chain_id="A") -> None: 
    seq, coords, scores = seqres_aligned_coords(INPUT + name + ".cif", chain_id)
    sequence=np.array(seq, dtype="U")
    np.savez_compressed(f"{OUT}/{name}_{chain_id}.npz", coords=coords, scores=scores, sequence=sequence)


with open("/n/groups/marks/projects/viral_plm/models/PoET-2/data/structures/pdb_acc.txt", 'r') as my_file: 
    for line in my_file: 
        protein, chain = line.split("_")
        prot = protein.strip()
        ch = chain.strip()
        print(chain)
        process_and_save(prot, ch)


