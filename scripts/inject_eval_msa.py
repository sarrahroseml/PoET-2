"""Inject MSA paths from eval MSA search into eval context fold JSONs.

The eval context sequences use ctx_HASH_NAME format. Their MSA results
from colabfold_search use the same naming (since the FASTA headers are
the ctx_HASH_NAME IDs).
"""

import json
import os
from pathlib import Path


def main():
    msa_dir = Path("data/eval_msa_jobs/outputs")
    input_dir = Path("data/eval_context_fold/inputs")
    paired_dir = input_dir / "paired_a3ms"
    paired_dir.mkdir(exist_ok=True)

    a3m_map = {}
    for batch_dir in sorted(msa_dir.glob("batch_*")):
        for a3m_file in batch_dir.glob("*.a3m"):
            full_stem = a3m_file.stem
            a3m_path = str(a3m_file.resolve())
            a3m_map[full_stem] = a3m_path
            short_id = full_stem.split("_sources_")[0] if "_sources_" in full_stem else full_stem
            if short_id not in a3m_map:
                a3m_map[short_id] = a3m_path

    print(f"A3M map: {len(a3m_map)} entries")

    found = 0
    missing = 0
    for json_file in sorted(input_dir.glob("batch_*.json")):
        with open(json_file) as f:
            entries = json.load(f)

        for entry in entries:
            name = entry["name"]
            seq = entry["sequences"][0]["proteinChain"]["sequence"]

            if name in a3m_map:
                entry["sequences"][0]["proteinChain"]["unpairedMsaPath"] = a3m_map[name]
                paired_path = str((paired_dir / f"{name}_paired.a3m").resolve())
                if not os.path.exists(paired_path):
                    with open(paired_path, "w") as pf:
                        pf.write(f">query\n{seq}\n")
                entry["sequences"][0]["proteinChain"]["pairedMsaPath"] = paired_path
                found += 1
            else:
                missing += 1

        with open(json_file, "w") as f:
            json.dump(entries, f, indent=2)

    print(f"Injected MSA: {found}, Missing: {missing}")


if __name__ == "__main__":
    main()
