import gemmi
doc = gemmi.cif.read("/n/groups/marks/projects/viral_plm/models/PoET-2/data/structures/pdb_files/pdb/8gpi.cif").sole_block()

# chains in mapping table (SEQRES/label)
scheme = doc.find(["_pdbx_poly_seq_scheme.asym_id"])
print("scheme asym_ids:", sorted({r[0] for r in scheme}))

# chains in atom table: label vs author
atom_cols = doc.find(["_atom_site.label_asym_id","_atom_site.auth_asym_id"])
lbl = [r[0] for r in atom_cols]
auth = [r[1] for r in atom_cols]
print("unique label_asym_id:", sorted(set(lbl)))
print("unique auth_asym_id :", sorted(set(auth)))