from Bio import SeqIO 
import pandas as pd
import numpy as np
struc_array_folder = "/n/groups/marks/projects/viral_plm/models/PoET-2/data/structures/structure_arrays"
TRAIN = "/n/groups/marks/projects/viral_plm/models/PoET-2/data/train_set"
import dask.dataframe as dd


df = dd.read_parquet(
    '/n/groups/marks/projects/viral_plm/models/PoET-2/data/structures/valid_structure_assignments.parquet',
)
pdf = df[["uniref90", "assigned_structure"]].compute()

def split_assigned(x: str):
    if x.startswith("SARRAHAF2"):
        prefix = "SARRAHAF2"
        rest   = x.split("-")[1]
    elif x.startswith("SARRAHV3D"):
        prefix = "SARRAHV3D"
        rest   = x.split("-")[1].split(".")[0]
    elif x.startswith("SARRAHVAF"):
        prefix = "SARRAHVAF"
        rest   = x[9:].split(".")[0]
    else:
        prefix = ""
        rest   = x[9:]
    return pd.Series([prefix, rest])

pdf[["prefix", "suffix"]] = pdf["assigned_structure"].apply(split_assigned)
mapping = pdf.set_index("uniref90")["suffix"].to_dict()


def _np_array_to_str(arr: np.ndarray) -> str:
    if isinstance(arr, np.ndarray):
        if arr.ndim == 0:
            seq = arr.item()
        elif arr.ndim == 1:
            seq = "".join(arr.tolist())
        else:
            raise ValueError(f"Unexpected sequence array shape {arr.shape}")
    else:
        seq = arr
    if isinstance(seq, (bytes, bytearray, np.bytes_)):
        seq = seq.decode("utf-8")
    return seq


for rec in SeqIO.parse("/n/groups/marks/projects/viral_plm/models/PoET-2/data/all_sequences.fasta", "fasta"):
    u90_acc = str(rec.id)
    
    # Only use sequences with structure
    if u90_acc not in mapping: 
        continue

    sequence = str(rec.seq) 
    struc = mapping[u90_acc]
    my_struc_path = f"{struc_array_folder}/{struc}.npz"

    struc_array = np.load(my_struc_path)
    struc_coords = struc_array["coords"]        # shape (L_struct, 3, 3)
    struc_sequence = _np_array_to_str(struc_array["sequence"])
    struc_scores = struc_array["scores"]        # shape (L_struct,)

    # Split original sequence into [$][structured][*][rest]
    first_seq = sequence.split("$", 1)[1].split("*", 1)[0]
    rest = sequence.split("*", 1)[1]

    final_sequence = "$" + struc_sequence + "*" + rest
    N_seq = len(final_sequence)

    # Allocate full-length arrays, including slots for $ and *
    my_coords = np.full((N_seq, 3, 3), np.nan, dtype=np.float32)
    my_scores = np.full((N_seq,), np.nan, dtype=np.float32)

    L_struct = len(struc_sequence)

    # Fill structured region *after* the '$' and *before* the '*'
    my_coords[1:1 + L_struct] = struc_coords
    my_scores[1:1 + L_struct] = struc_scores

    np.savez_compressed(
        f"{TRAIN}/{u90_acc}.npz",
        coords=my_coords,
        scores=my_scores,
        sequence=final_sequence,
    )