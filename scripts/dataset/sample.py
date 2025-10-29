"""Utility for traversing UniRef clusters via the UniProt REST API.

Given a UniRef90 cluster accession, fetches its parent UniRef50
cluster and then enumerates all UniRef90 clusters that belong to that
UniRef50 cluster.
"""

from __future__ import annotations

import argparse
import random
import sys
import threading
import atexit
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
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
_session_local = threading.local()
_cache_lock = threading.Lock()
CACHE_FILE = Path.home() / ".poet_uniref_cache.json"
MISSING_CLUSTERS_PATH = Path("missing_uniref_clusters.txt")

_uniref50_members_cache: Dict[str, List[str]] = {}
_uniref90_members_cache: Dict[str, List[str]] = {}
_cache_dirty = False

if CACHE_FILE.exists():
    try:
        cache_data = json.loads(CACHE_FILE.read_text())
        _uniref50_members_cache = {
            key: list(value) for key, value in cache_data.get("uniref50_members", {}).items()
        }
        _uniref90_members_cache = {
            key: list(value) for key, value in cache_data.get("uniref90_members", {}).items()
        }
    except (OSError, ValueError, TypeError):
        _uniref50_members_cache = {}
        _uniref90_members_cache = {}


def _persist_cache() -> None:
    if not _cache_dirty:
        return
    data = {
        "uniref50_members": _uniref50_members_cache,
        "uniref90_members": _uniref90_members_cache,
    }
    try:
        CACHE_FILE.write_text(json.dumps(data))
    except OSError:
        pass


atexit.register(_persist_cache)


def _log_missing_cluster(cluster_id: str, message: str) -> None:
    line = f"{cluster_id}\t{message}\n"
    with _cache_lock:
        try:
            with MISSING_CLUSTERS_PATH.open("a", encoding="utf-8") as handle:
                handle.write(line)
        except OSError:
            pass


def _thread_session() -> requests.Session:
    session = getattr(_session_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        _session_local.session = session
    return session


class UniRefError(RuntimeError):
    """Custom error raised for UniRef lookup failures."""


@dataclass
class UniRef90ClusterInfo:
    """Aggregated data for a UniRef90 cluster."""
    cluster_id: str
    uniref100_ids: List[str]

    @property
    def size(self) -> int:
        return len(self.uniref100_ids)


@dataclass(frozen=True)
class SamplingRecord:
    """Probability-bearing record for a UniRef100 sequence."""

    uniref90_id: str
    uniref90_size: int
    uniref100_id: str
    probability: float


def _request_json(url: str, params: Optional[Dict[str, str]] = None) -> Dict:
    """Perform a GET request and return the JSON payload."""
    try:
        response = _thread_session().get(url, params=params, timeout=REQUEST_TIMEOUT)
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
    with _cache_lock:
        cached = _uniref50_members_cache.get(normalized_id)
    if cached is not None:
        return list(cached)

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
    result = sorted(seen)
    with _cache_lock:
        global _cache_dirty
        _uniref50_members_cache[normalized_id] = list(result)
        _cache_dirty = True
    return result


def collect_uniref90_cluster_info(uniref90_id: str) -> UniRef90ClusterInfo:
    """Gather UniRef100 sequence data for the provided UniRef90 cluster."""
    normalized_id = _normalize_cluster_id(uniref90_id, "UniRef90")
    with _cache_lock:
        cached = _uniref90_members_cache.get(normalized_id)
    if cached is not None:
        return UniRef90ClusterInfo(
            cluster_id=normalized_id, uniref100_ids=list(cached)
        )

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

    sorted_candidates = sorted(uniref100_candidates)
    with _cache_lock:
        global _cache_dirty
        _uniref90_members_cache[normalized_id] = list(sorted_candidates)
        _cache_dirty = True

    return UniRef90ClusterInfo(
        cluster_id=normalized_id, uniref100_ids=sorted_candidates
    )


def gather_uniref90_clusters_parallel(
    cluster_ids: Sequence[str], max_workers: int = 4
) -> Tuple[List[UniRef90ClusterInfo], List[Tuple[str, str]]]:
    clusters: List[UniRef90ClusterInfo] = []
    errors: List[Tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(collect_uniref90_cluster_info, cluster_id): cluster_id
            for cluster_id in cluster_ids
        }
        for future in as_completed(future_map):
            cluster_id = future_map[future]
            try:
                clusters.append(future.result())
            except UniRefError as exc:
                errors.append((cluster_id, str(exc)))
                _log_missing_cluster(cluster_id, str(exc))
    return clusters, errors


def build_sampling_pool(
    seed_uniref90_id: str, max_workers: int = 4
) -> Tuple[str, List[UniRef90ClusterInfo]]:
    """Assemble sampling-ready data for the supplied UniRef90 accession."""
    uniref50_id = infer_uniref50_id(seed_uniref90_id)
    uniref90_list = collect_uniref90_members(uniref50_id)
    uniref90_clusters, errors = gather_uniref90_clusters_parallel(
        uniref90_list, max_workers=max_workers
    )
    for cluster_id, msg in errors:
        print(f"Warning: failed to load {cluster_id}: {msg}", file=sys.stderr)
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
        for u100 in cluster.uniref100_ids:
            records.append(
                SamplingRecord(
                    uniref90_id=cluster.cluster_id,
                    uniref90_size=cluster.size,
                    uniref100_id=u100,
                    probability=sequence_weight,
                )
            )
    return records


def write_uniref100_accession(
    records: Sequence[SamplingRecord], destination: Path
) -> None:
    """Write UniRef100 accessions to a txt file with probability metadata."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="ascii") as handle:
        for record in records:
            header = (
                f"{record.uniref100_id} "
                f"uniref90={record.uniref90_id} "
                f"probability={record.probability:.10f} "
                f"uniref90_size={record.uniref90_size}"
            )
            handle.write(header + "\n")


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
        "--out-path",
        type=Path,
        default=None,
        help="Optional path to write sampled UniRef100 accessions.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads",
    )

    args = parser.parse_args(argv)
    if args.sample_size is not None and args.sample_size < 0:
        parser.error("--sample-size must be non-negative.")

    try:
        # Return Uniref90 clusters of the parent Uniref50 cluster
        uniref50_id, clusters = build_sampling_pool(args.uniref90_id, args.workers)
        records = compute_sampling_records(clusters)
    except UniRefError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        

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
        for uniref100_id in cluster.uniref100_ids:
            record = record_lookup[(cluster.cluster_id, uniref100_id)]
            print(f"  - {uniref100_id} | probability={record.probability:.6f}")

    if args.sample_size:
        rng = random.Random(args.seed)
        weights = [record.probability for record in records]
        sampled_records = rng.choices(records, weights=weights, k=args.sample_size)
        if args.out_path:
            write_uniref100_accession(sampled_records, args.out_path)
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
