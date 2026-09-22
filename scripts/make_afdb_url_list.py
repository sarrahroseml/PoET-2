"""Generate a URL list for bulk AFDB downloads from the extracted ID list.

Reads all_ids.txt (one FASTA ID per line) and outputs a TSV of
(accession, url) for wget/xargs parallel download.

Usage:
    python3 scripts/make_afdb_url_list.py \
        --ids data/structure_ids/all_ids.txt \
        --out data/structure_ids/afdb_urls.tsv
"""

import argparse
import sys

AFDB_V6 = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v6.pdb"


def extract_accession(seq_id):
    if seq_id.endswith("|af2"):
        bare = seq_id[:seq_id.rfind("|")]
        if bare.startswith("AF-") and "-F1-" in bare:
            return bare.split("-")[1]
        return bare

    if seq_id.endswith("|bfvd"):
        return seq_id[:seq_id.rfind("|")]

    if seq_id.startswith("tr|") or seq_id.startswith("sp|"):
        parts = seq_id.split("|")
        if len(parts) >= 2:
            return parts[1]

    if seq_id.startswith("UniRef100_"):
        return seq_id[len("UniRef100_"):]

    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    seen = set()
    count = 0
    with open(args.ids) as fin, open(args.out, "w") as fout:
        for line in fin:
            seq_id = line.strip()
            if not seq_id:
                continue
            acc = extract_accession(seq_id)
            if acc is None or acc in seen:
                continue
            seen.add(acc)
            fout.write("%s\t%s\n" % (acc, AFDB_V6.format(acc=acc)))
            count += 1

    print("Wrote %d unique accessions to %s" % (count, args.out))


if __name__ == "__main__":
    main()
