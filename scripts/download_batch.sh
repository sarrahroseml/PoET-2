#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: $(basename "$0") -i input.txt [-o output_dir] [-C cache_dir] [--overwrite]

  -i FILE   Input file: one PDB id or PDB id + chain per line (e.g. 7x25 or 7x25_G)
  -o DIR    Directory for extracted chain files (default: ./chains)
  -C DIR    Directory to cache full PDB downloads (default: ./pdb_cache)
  --overwrite  Recreate chain files even if they already exist
USAGE
}

INPUT=""
OUT_DIR="chains"
CACHE_DIR="pdb_cache"
OVERWRITE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input)
      INPUT="$2"
      shift 2
      ;;
    -o|--output)
      OUT_DIR="$2"
      shift 2
      ;;
    -C|--cache)
      CACHE_DIR="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$INPUT" ]]; then
  echo "Error: input file required (-i)." >&2
  usage >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl is required." >&2
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "Error: jq is required." >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$CACHE_DIR"

RCSB_ENTRY_API="https://data.rcsb.org/rest/v1/core/entry"
RCSB_BASE="https://files.rcsb.org/download"

resolve_chain() {
  local entry="$1" desired="$2"
  if [[ -z "$desired" ]]; then
    echo ""
    return
  fi
  local json
  json=$(curl -fsS "$RCSB_ENTRY_API/${entry^^}" 2>/dev/null || true)
  if [[ -z "$json" ]]; then
    echo "$desired"
    return
  fi
  local resolved
  resolved=$(
  echo "$json" |
  jq -r --arg chain "$desired" '
    [.struct_asym // [] | .[] 
      | select((.auth_asym_id // "") == $chain or (.label_asym_id // "") == $chain)
      | .auth_asym_id][0]
  '
)
  if [[ -n "$resolved" && "$resolved" != "null" ]]; then
    echo "$resolved"
  else
    echo "$desired"
  fi
}

ensure_pdb_cached() {
  local entry="$1"
  local pdb_path="$CACHE_DIR/${entry}.pdb"
  local cif_path="$CACHE_DIR/${entry}.cif"

  if [[ -f "$pdb_path" ]]; then
    echo "$pdb_path"
    return 0
  fi

  local url="$RCSB_BASE/${entry}.pdb"
  echo "Fetching $url" >&2
  if curl -fsS "$url" -o "$pdb_path"; then
    echo "$pdb_path"
    return 0
  fi
  rm -f "$pdb_path"

  url="$RCSB_BASE/${entry}.cif"
  echo "Fetching $url" >&2
  if ! curl -fsS "$url" -o "$cif_path"; then
    echo "Failed to download $entry as PDB or CIF" >&2
    rm -f "$cif_path"
    return 1
  fi

  PYTHON_BIN=${PYTHON:-python3}
  if ! "$PYTHON_BIN" -c "import gemmi" >/dev/null 2>&1; then
    echo "Error: gemmi (Python module) is required to convert CIF for $entry" >&2
    return 1
  fi

  if ! "$PYTHON_BIN" - "$cif_path" "$pdb_path" <<'PY'
import sys
import gemmi

cif_path, pdb_path = sys.argv[1:3]
structure = gemmi.read_structure(cif_path)
structure.write_pdb(pdb_path)
PY
  then
    echo "Failed to convert CIF to PDB for $entry" >&2
    rm -f "$pdb_path"
    return 1
  fi

  echo "$pdb_path"
  return 0
}

extract_chain() {
  local pdb_file="$1" chain_id="$2" out_file="$3"
  if [[ -z "$chain_id" ]]; then
    cp "$pdb_file" "$out_file"
    return
  fi
  awk -v chain="$chain_id" '
    BEGIN { len = length(chain); }
    /^ATOM  / || /^HETATM/ || /^ANISOU/ {
      if (substr($0, 22, len) == chain) print
      next
    }
    /^TER / {
      if (substr($0, 22, len) == chain) print
      next
    }
    /^END/ { print }
  ' "$pdb_file" > "$out_file"
}

while IFS= read -r raw || [[ -n "$raw" ]]; do
  line=$(echo "$raw" | tr -d ' \t\r')
  if [[ -z "$line" || "$line" =~ ^# ]]; then
    continue
  fi
  entry=${line%%_*}
  entry=${entry^^}
  chain=""
  if [[ "$line" == *"_"* ]]; then
    chain=${line#*_}
    chain=${chain^^}
  fi

  pdb_file=$(ensure_pdb_cached "$entry") || {
    echo "Skipping $entry due to download/convert failure." >&2
    continue
  }

  actual_chain="$chain"
  if [[ -n "$chain" ]]; then
    actual_chain=$(resolve_chain "$entry" "$chain")
    if [[ -z "$actual_chain" ]]; then
      echo "Could not resolve chain $chain for $entry; skipping." >&2
      continue
    fi
  fi

  out_file="$OUT_DIR/${entry}"
  if [[ -n "$chain" ]]; then
    out_file+="_${chain}"
  fi
  out_file+=".pdb"

  if [[ -f "$out_file" && $OVERWRITE == false ]]; then
    echo "$out_file already exists; skipping."
    continue
  fi

  extract_chain "$pdb_file" "$actual_chain" "$out_file"

  if [[ ! -s "$out_file" ]]; then
    echo "Warning: extracted file $out_file is empty (chain $actual_chain)." >&2
    rm -f "$out_file"
  else
    echo "Wrote $out_file"
  fi

done < "$INPUT"
