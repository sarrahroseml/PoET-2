from __future__ import annotations
from Bio import SeqIO

import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

TOKEN_LIMIT = 8192
H_SEQS_DIR = Path("/n/groups/marks/projects/viral_plm/models/PoET-2/data/h_seqs")
SAMPLE_LOG_PATH = Path("/n/groups/marks/projects/viral_plm/models/PoET-2/data/sample_log.txt")
OUT_FILE = Path("/n/groups/marks/projects/viral_plm/models/PoET-2/data/valid_sequences.fasta")

S = "$"
E = "*"

BASE_URL = "https://rest.uniprot.org/uniref"
REQUEST_TIMEOUT = 30
USER_AGENT = "PoET-2-homolog-downloader/0.1"
my_seq_dict = SeqIO.index(str(OUT_FILE), "fasta")

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": USER_AGENT})
_SESSION.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
        )
    ),
)


def _normalize_uniref_id(prefix: str, accession: str) -> str:
    accession = accession.strip()
    if not accession:
        raise ValueError("UniRef accession cannot be empty")
    if accession.startswith(prefix + "_"):
        return accession
    if accession.startswith(prefix):
        return accession
    if accession.startswith("UniRef"):
        return f"{prefix}_{accession.split('_', 1)[-1]}"
    return f"{prefix}_{accession}"





def sample_until_limit(uniref90_acc: str) -> None:
    base_sequence = fetch_uniref90_sequence(uniref90_acc)
    if not base_sequence:
        print(f"Warning: no sequence found for {uniref90_acc}; skipping", file=sys.stderr)
        return

    token_limit = TOKEN_LIMIT
    assembled = [S, base_sequence, E]
    token_limit -= len(base_sequence) + 2

    sample_seqs = 0
    file_path = (H_SEQS_DIR / uniref90_acc).with_suffix(".txt")

    try:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                uid = line.split()[0]
                seq = fetch_uniref100_sequence(uid)
                if not seq:
                    continue

                needed = len(seq) + 2  # start/end tokens
                if token_limit - needed < 1:
                    available = token_limit - 1 - 2  # leave room for final '*'
                    if available > 0:
                        assembled.extend((S, seq[:available], E))
                        sample_seqs += 1
                    break

                assembled.extend((S, seq, E))
                token_limit -= needed
                sample_seqs += 1
    except FileNotFoundError:
        print(f"Warning: homolog file missing for {uniref90_acc}: {file_path}", file=sys.stderr)
        return

    assembled_str = "".join(assembled)
    if len(assembled_str) >= TOKEN_LIMIT:
        assembled_str = assembled_str[: TOKEN_LIMIT - 1] + E

    SAMPLE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SAMPLE_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{uniref90_acc} {sample_seqs}\n")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("a", encoding="utf-8") as fout:
        fout.write(f">{uniref90_acc}\n")
        fout.write(assembled_str + "\n")


@lru_cache(maxsize=None)
def _fetch_uniref_sequence(accession: str) -> Optional[str]:
    params = {"format": "fasta"}
    try:
        resp = _SESSION.get(accession, params=params, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        print(f"Error fetching {accession}: {exc}", file=sys.stderr)
        return None
    if resp.status_code == 404:
        return None
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        print(f"HTTP error for {accession}: {exc}", file=sys.stderr)
        return None

    seq = "".join(line.strip() for line in resp.text.splitlines() if not line.startswith(">"))
    return seq or None


def fetch_uniref90_sequence(uniref90_id: str) -> Optional[str]:
    cluster = _normalize_uniref_id("UniRef90", uniref90_id)
    return _fetch_uniref_sequence(f"{BASE_URL}/{cluster}")


def fetch_uniref100_sequence(uniref100_id: str) -> Optional[str]:
    cluster = _normalize_uniref_id("UniRef100", uniref100_id)
    return _fetch_uniref_sequence(f"{BASE_URL}/{cluster}")


for item_path in H_SEQS_DIR.iterdir():
    pass
    if item_path.name.endswith(".txt"):
        ides = item_path.name[:-4]
        if ides in my_seq_dict: 
            print(f"Seem {ides} already, skipping")
            continue

        print(f"Processing {ides}")
        sample_until_limit(ides)
