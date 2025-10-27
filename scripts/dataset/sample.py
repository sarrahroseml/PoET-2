"""Utility for traversing UniRef clusters via the UniProt REST API.

Given a UniRef90 cluster accession, fetches its parent UniRef50
cluster and then enumerates all UniRef90 clusters that belong to that
UniRef50 cluster.
"""

from __future__ import annotations

import argparse
import random
import sys
import textwrap
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import urljoin

import requests

BASE_URL = "https://rest.uniprot.org/uniref"
REQUEST_TIMEOUT = 30
USER_AGENT = (
    "PoET-2-UniRefHelper/0.1 (+https://github.com/OpenProteinAI/PoET-2)"
)

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": USER_AGENT})


class UniRefError(RuntimeError):
    """Custom error raised for UniRef lookup failures."""


@dataclass(frozen=True)
class UniRef100Sequence:
    """Container for UniRef100 sequence data."""

    cluster_id: str
    sequence: str
    length: Optional[int]
    accessions: Sequence[str]
    cluster_member_count: int


@dataclass
class UniRef90ClusterInfo:
    """Aggregated data for a UniRef90 cluster."""

    cluster_id: str
    uniref100_sequences: List[UniRef100Sequence]

    @property
    def size(self) -> int:
        return len(self.uniref100_sequences)


@dataclass(frozen=True)
class SamplingRecord:
    """Probability-bearing record for a UniRef100 sequence."""

    uniref90_id: str
    uniref90_size: int
    uniref100_id: str
    sequence: str
    length: Optional[int]
    uniref100_cluster_size: int
    probability: float


def _request_json(url: str, params: Optional[Dict[str, str]] = None) -> Dict:
    """Perform a GET request and return the JSON payload."""
    try:
        response = _SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        raise UniRefError(f"Network error while querying {url}: {exc}") from exc

    if response.status_code == 404:
        raise UniRefError(f"No UniRef cluster found for {url}")

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        snippet = response.text[:200].replace("\n", " ")
        raise UniRefError(
            f"UniProt API returned HTTP {response.status_code}: {snippet}"
        ) from exc

    try:
        return response.json()
    except ValueError as exc:
        raise UniRefError("Failed to parse UniProt response as JSON") from exc


def _normalize_cluster_id(cluster_id: str, prefix: str) -> str:
    """Ensure the cluster id has the expected UniRef prefix."""
    trimmed = cluster_id.strip()
    if not trimmed:
        raise UniRefError("Cluster accession cannot be empty.")

    if "_" in trimmed:
        prefix_part, suffix = trimmed.split("_", 1)
        if prefix_part.lower() != prefix.lower():
            raise UniRefError(
                f"Expected a {prefix} accession but received '{cluster_id}'."
            )
    else:
        suffix = trimmed

    return f"{prefix}_{suffix.upper()}"


def fetch_uniref_cluster_page(
    cluster_id: str, *, size: int = 500, include_sequences: bool = False
) -> Dict:
    """Retrieve a UniRef cluster page."""
    params = {"format": "json", "size": str(size)}
    if include_sequences:
        params["include"] = "sequences"
    url = f"{BASE_URL}/{cluster_id}"
    return _request_json(url, params=params)


def collect_full_cluster(
    cluster_id: str, *, include_sequences: bool = False
) -> Tuple[Dict, List[Dict]]:
    """Collect representative data and all members, following pagination."""
    data = fetch_uniref_cluster_page(
        cluster_id, include_sequences=include_sequences
    )
    representative = data.get("representativeMember", {})
    members = list(data.get("members", []))

    next_url = data.get("nextPageUrl") or _extract_next_link(data)
    while next_url:
        abs_url = (
            next_url
            if next_url.startswith("http")
            else urljoin(f"{BASE_URL}/", next_url)
        )
        payload = _request_json(abs_url)
        members.extend(payload.get("members", []))
        next_url = payload.get("nextPageUrl") or _extract_next_link(payload)

    return representative, members


def _extract_next_link(payload: Dict) -> Optional[str]:
    """Extract the next page URL from the payload if present."""
    for link in payload.get("links", []):
        if link.get("rel") == "next" and link.get("href"):
            return link["href"]
    return None


def infer_uniref50_id(uniref90_id: str) -> str:
    """Determine the UniRef50 cluster that contains the supplied UniRef90."""
    normalized_id = _normalize_cluster_id(uniref90_id, "UniRef90")
    data = fetch_uniref_cluster_page(normalized_id)

    candidates: List[Optional[str]] = []
    representative = data.get("representativeMember") or {}
    candidates.append(representative.get("uniref50Id"))
    candidates.extend(member.get("uniref50Id") for member in data.get("members", []))

    for candidate in candidates:
        if candidate:
            return candidate

    raise UniRefError(
        f"Could not determine a UniRef50 cluster for {normalized_id}"
    )


def collect_uniref90_members(uniref50_id: str) -> List[str]:
    """Return all UniRef90 cluster accessions within the UniRef50 cluster."""
    normalized_id = _normalize_cluster_id(uniref50_id, "UniRef50")
    representative, members = collect_full_cluster(normalized_id)

    seen: Set[str] = set()
    candidate = representative.get("uniref90Id")
    if candidate:
        seen.add(candidate)

    for member in members:
        candidate = member.get("uniref90Id")
        if candidate:
            seen.add(candidate)

    if not seen:
        raise UniRefError(
            f"UniRef50 cluster {normalized_id} lists no UniRef90 members."
        )

    return sorted(seen)


@lru_cache(maxsize=None)
def fetch_uniref100_cluster(uniref100_id: str) -> Tuple[Dict, List[Dict]]:
    """Fetch a UniRef100 cluster with cached results."""
    normalized_id = _normalize_cluster_id(uniref100_id, "UniRef100")
    return collect_full_cluster(normalized_id, include_sequences=True)


def _sequence_from_entry(entry: Dict) -> Tuple[Optional[str], Optional[int]]:
    """Extract the sequence string and length from an entry if available."""
    sequence_block = entry.get("sequence") or {}
    value = sequence_block.get("value")
    length = sequence_block.get("length") or entry.get("sequenceLength")
    if value and not length:
        length = len(value)
    return value, length


def build_uniref100_sequence(uniref100_id: str) -> UniRef100Sequence:
    """Construct a UniRef100Sequence object for the provided cluster."""
    representative, members = fetch_uniref100_cluster(uniref100_id)
    sequence, length = _sequence_from_entry(representative)

    if not sequence:
        for member in members:
            sequence, length = _sequence_from_entry(member)
            if sequence:
                break

    if not sequence:
        raise UniRefError(
            f"UniRef100 cluster {uniref100_id} does not provide a sequence."
        )

    accessions: Sequence[str] = representative.get("accessions") or ()
    cluster_member_count = 1 + len(members)

    return UniRef100Sequence(
        cluster_id=_normalize_cluster_id(uniref100_id, "UniRef100"),
        sequence=sequence,
        length=length,
        accessions=tuple(accessions),
        cluster_member_count=cluster_member_count,
    )


def collect_uniref90_cluster_info(uniref90_id: str) -> UniRef90ClusterInfo:
    """Gather UniRef100 sequence data for the provided UniRef90 cluster."""
    normalized_id = _normalize_cluster_id(uniref90_id, "UniRef90")
    representative, members = collect_full_cluster(normalized_id)

    uniref100_candidates: Set[str] = set()
    rep_candidate = representative.get("uniref100Id")
    if rep_candidate:
        uniref100_candidates.add(rep_candidate)

    for member in members:
        candidate = member.get("uniref100Id")
        if candidate:
            uniref100_candidates.add(candidate)

    if not uniref100_candidates:
        raise UniRefError(
            f"UniRef90 cluster {normalized_id} contains no UniRef100 members."
        )

    sequences = [
        build_uniref100_sequence(candidate) for candidate in sorted(uniref100_candidates)
    ]

    return UniRef90ClusterInfo(
        cluster_id=normalized_id, uniref100_sequences=sequences
    )


def build_sampling_pool(
    seed_uniref90_id: str,
) -> Tuple[str, List[UniRef90ClusterInfo]]:
    """Assemble sampling-ready data for the supplied UniRef90 accession."""
    uniref50_id = infer_uniref50_id(seed_uniref90_id)
    uniref90_clusters = [
        collect_uniref90_cluster_info(cluster_id)
        for cluster_id in collect_uniref90_members(uniref50_id)
    ]
    return uniref50_id, uniref90_clusters


def compute_sampling_records(
    clusters: Sequence[UniRef90ClusterInfo],
) -> List[SamplingRecord]:
    """Compute sampling probabilities for UniRef100 sequences."""
    if not clusters:
        raise UniRefError("No UniRef90 clusters were found for sampling.")

    total_clusters = len(clusters)
    cluster_weight = 1.0 / total_clusters
    records: List[SamplingRecord] = []

    for cluster in clusters:
        if cluster.size == 0:
            raise UniRefError(
                f"UniRef90 cluster {cluster.cluster_id} has no UniRef100 members."
            )
        sequence_weight = cluster_weight / cluster.size
        for sequence in cluster.uniref100_sequences:
            records.append(
                SamplingRecord(
                    uniref90_id=cluster.cluster_id,
                    uniref90_size=cluster.size,
                    uniref100_id=sequence.cluster_id,
                    sequence=sequence.sequence,
                    length=sequence.length,
                    uniref100_cluster_size=sequence.cluster_member_count,
                    probability=sequence_weight,
                )
            )

    return records


def _sequence_token_count(record: SamplingRecord) -> int:
    """Return the token count for a sampled sequence (AA residues)."""
    if record.length:
        return record.length
    return len(record.sequence)


def sample_until_token_limit(
    records: Sequence[SamplingRecord],
    token_limit: int,
    rng: random.Random,
) -> Tuple[List[SamplingRecord], int]:
    """Sample records until the cumulative token limit is reached."""
    if token_limit <= 0:
        raise UniRefError("--token-limit must be a positive integer.")
    weights = [record.probability for record in records]
    total_tokens = 0
    sampled: List[SamplingRecord] = []

    while total_tokens < token_limit:
        record = rng.choices(records, weights=weights, k=1)[0]
        tokens = _sequence_token_count(record)
        if tokens <= 0:
            raise UniRefError(
                f"Sequence {record.uniref100_id} has non-positive token count."
            )
        sampled.append(record)
        total_tokens += tokens

    return sampled, total_tokens


def write_fasta(records: Sequence[SamplingRecord], destination: Path) -> None:
    """Write UniRef100 sequences to a FASTA file with probability metadata."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="ascii") as handle:
        for record in records:
            header = (
                f">{record.uniref100_id} "
                f"uniref90={record.uniref90_id} "
                f"probability={record.probability:.10f} "
                f"uniref90_size={record.uniref90_size} "
                f"uniref100_size={record.uniref100_cluster_size}"
            )
            handle.write(header + "\n")
            for line in textwrap.wrap(record.sequence, width=60):
                handle.write(line + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve the UniRef50 cluster corresponding to a UniRef90 cluster "
            "and enumerate the UniRef90 and UniRef100 members needed for sampling."
        )
    )
    parser.add_argument(
        "uniref90_id",
        help="UniRef90 cluster accession (e.g. 'UniRef90_Q9H9K5' or 'Q9H9K5').",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help=(
            "Number of UniRef100 sequences to sample according to the derived "
            "probabilities. Sampling is performed with replacement."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed used when sampling.",
    )
    parser.add_argument(
        "--token-limit",
        type=int,
        default=0,
        help=(
            "Sample UniRef100 sequences until the cumulative residue count "
            "reaches this limit. Incompatible with --sample-size."
        ),
    )
    parser.add_argument(
        "--fasta-out",
        type=Path,
        default=None,
        help="Optional path to write the UniRef100 sequences in FASTA format.",
    )

    args = parser.parse_args(argv)
    if args.sample_size is not None and args.sample_size < 0:
        parser.error("--sample-size must be non-negative.")
    if args.token_limit is not None and args.token_limit < 0:
        parser.error("--token-limit must be non-negative.")
    if args.sample_size and args.token_limit:
        parser.error("--sample-size and --token-limit are mutually exclusive.")

    try:
        uniref50_id, clusters = build_sampling_pool(args.uniref90_id)
        records = compute_sampling_records(clusters)
    except UniRefError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    total_sequences = len(records)
    print(f"UniRef50 parent cluster: {uniref50_id}")
    print(f"UniRef90 clusters in {uniref50_id}: {len(clusters)}")
    print(f"Total UniRef100 sequences across clusters: {total_sequences}")

    record_lookup = {
        (record.uniref90_id, record.uniref100_id): record for record in records
    }
    cluster_weight = 1.0 / len(clusters)

    for cluster in clusters:
        print()
        print(
            f"{cluster.cluster_id}: {cluster.size} UniRef100 clusters "
            f"(cluster weight {cluster_weight:.6f})"
        )
        for sequence in cluster.uniref100_sequences:
            record = record_lookup[(cluster.cluster_id, sequence.cluster_id)]
            accession_str = ", ".join(sequence.accessions) or "n/a"
            length_str = str(sequence.length) if sequence.length else "unknown"
            print(
                f"  - {sequence.cluster_id} | length={length_str} | "
                f"UniRef100 members={sequence.cluster_member_count} | "
                f"probability={record.probability:.6f}"
            )
            if accession_str != "n/a":
                print(f"    accessions: {accession_str}")
            wrapped_sequence = textwrap.wrap(sequence.sequence, width=80)
            for line in wrapped_sequence:
                print(f"    {line}")

    if args.fasta_out:
        write_fasta(records, args.fasta_out)
        print()
        print(f"FASTA written to {args.fasta_out}")

    sampled_records: List[SamplingRecord] = []
    if args.token_limit:
        rng = random.Random(args.seed)
        sampled_records, used_tokens = sample_until_token_limit(
            records, args.token_limit, rng
        )
        print()
        print(
            f"Sampled {len(sampled_records)} sequences to reach "
            f"{used_tokens} tokens (limit {args.token_limit})."
        )
        for index, record in enumerate(sampled_records, 1):
            print(
                f"{index}. {record.uniref100_id} "
                f"(UniRef90={record.uniref90_id}, tokens={_sequence_token_count(record)}, "
                f"prob={record.probability:.6f})"
            )
    elif args.sample_size:
        rng = random.Random(args.seed)
        weights = [record.probability for record in records]
        sampled_records = rng.choices(records, weights=weights, k=args.sample_size)
        print()
        print(f"Sampled {args.sample_size} UniRef100 sequences:")
        for index, record in enumerate(sampled_records, 1):
            print(
                f"{index}. {record.uniref100_id} "
                f"(UniRef90={record.uniref90_id}, prob={record.probability:.6f})"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
