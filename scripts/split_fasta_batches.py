"""Split a FASTA file into batches for parallel colabfold_search."""
import argparse
from pathlib import Path


def split_fasta(fasta_path: str, out_dir: str, batch_size: int):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    batch_idx = 0
    seq_count = 0
    current_lines = []

    def flush():
        nonlocal batch_idx, current_lines
        if not current_lines:
            return
        (out / f"batch_{batch_idx}.fasta").write_text("".join(current_lines))
        batch_idx += 1
        current_lines = []

    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                if seq_count > 0 and seq_count % batch_size == 0:
                    flush()
                seq_count += 1
            current_lines.append(line)
    flush()

    print(f"Split {seq_count} sequences into {batch_idx} batches of ~{batch_size}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fasta", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=1000)
    args = p.parse_args()
    split_fasta(args.fasta, args.out_dir, args.batch_size)
