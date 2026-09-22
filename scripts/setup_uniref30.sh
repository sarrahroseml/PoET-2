#!/bin/bash
# Set up UniRef30 database for colabfold_search after downloading.
# Run on a compute node with enough memory for indexing.
#
# Steps: extract tar.gz, build mmseqs profile db, create search index,
# download taxonomy mapping.

set -euo pipefail

DB_DIR="${1:?Usage: $0 <db_dir>}"
cd "$DB_DIR"

export PATH="/n/netscratch/marks_lab/Lab/nyoussef/final_poet/PoET-2/tools/mmseqs/bin:$PATH"
export MMSEQS_FORCE_MERGE=1

UNIREF30DB="uniref30_2302"

echo "=== Extracting $UNIREF30DB.tar.gz ==="
echo "Start: $(date)"
tar -xzf "${UNIREF30DB}.tar.gz"
echo "Extracted at $(date)"

if [ -f "${UNIREF30DB}_db" ]; then
    echo "Prebuilt .db format detected — skipping tsv2exprofiledb"
else
    echo "=== Building MMseqs2 profile database ==="
    mmseqs tsv2exprofiledb "${UNIREF30DB}" "${UNIREF30DB}_db"
    echo "Profile DB built at $(date)"
fi

echo "=== Creating search index ==="
mmseqs createindex "${UNIREF30DB}_db" tmp1 --remove-tmp-files 1
echo "Index created at $(date)"

echo "=== Downloading taxonomy ==="
curl -sfL -o "${UNIREF30DB}_newtaxonomy.tar.gz" \
    "https://opendata.mmseqs.org/colabfold/uniref30_2302_newtaxonomy.tar.gz" || true
if [ -f "${UNIREF30DB}_newtaxonomy.tar.gz" ]; then
    tar -xzf "${UNIREF30DB}_newtaxonomy.tar.gz"
fi

if [ -e "${UNIREF30DB}_db_mapping" ]; then
    TAXHEADER=$(od -An -N4 -t x4 "${UNIREF30DB}_db_mapping" | tr -d ' ')
    if [ "${TAXHEADER}" != "0c170013" ]; then
        mmseqs createbintaxmapping "${UNIREF30DB}_db_mapping" "${UNIREF30DB}_db_mapping.bin"
        mv -f -- "${UNIREF30DB}_db_mapping.bin" "${UNIREF30DB}_db_mapping"
    fi
    ln -sf "${UNIREF30DB}_db_mapping" "${UNIREF30DB}_db.idx_mapping"
fi
if [ -e "${UNIREF30DB}_db_taxonomy" ]; then
    ln -sf "${UNIREF30DB}_db_taxonomy" "${UNIREF30DB}_db.idx_taxonomy"
fi

touch UNIREF30_READY
echo "=== All done at $(date) ==="
ls -lh "${UNIREF30DB}_db"*
