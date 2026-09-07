#!/usr/bin/env bash
# Fetch a FLEURS test split into data/ as a parquet the harness can read.
#
#   scripts/fetch-corpus.sh ha yo ig en
#
# Why this exists rather than letting `datasets` download:
#
#   * huggingface_hub's downloader stalled at 0 KB/s while plain HTTP to the
#     same URL sustained 1.2 MB/s, and its resume logic truncated a 674 MB
#     partial back to 494 MB and corrupted it.
#   * Append-only ranges cannot lose banked bytes. A failed chunk costs the
#     chunk, never the file — which `curl -C -` does not guarantee, because a
#     retry that fails to resume truncates and restarts.
#
# Idempotent: an already-complete file is left alone, a partial one continues.
set -uo pipefail

REV=70bb2e84b976b7e960aa89f1c648e09c59f894dd   # pinned; a moving ref is not reproducible
BASE="https://huggingface.co/datasets/google/fleurs/resolve/$REV/parquet-data"
ATTEMPTS=15

# A case statement, not an associative array: macOS ships bash 3.2, which has
# no `declare -A`, and the script failed there with "unbound variable".
config_for() {
  case "$1" in
    ha) echo ha_ng ;;
    yo) echo yo_ng ;;
    ig) echo ig_ng ;;
    en) echo en_us ;;
    *)  echo "" ;;
  esac
}

cd "$(dirname "$0")/.." || exit 1
mkdir -p data

for lang in "$@"; do
  cfg="$(config_for "$lang")"
  if [ -z "$cfg" ]; then
    echo "unknown language '$lang' — known: ha yo ig en" >&2
    continue
  fi

  out="data/${cfg}-test.parquet"
  url="$BASE/${cfg}/test-00000-of-00001.parquet"

  total=$(curl -sIL --max-time 60 "$url" | awk 'BEGIN{IGNORECASE=1} /^content-length:/{v=$2} END{gsub("\r","",v); print v}')
  if [ -z "$total" ] || [ "$total" -lt 1000 ]; then
    echo "$cfg: could not determine size — skipping" >&2
    continue
  fi

  have=$(stat -f%z "$out" 2>/dev/null || stat -c%s "$out" 2>/dev/null || echo 0)
  if [ "$have" -ge "$total" ]; then
    printf "%-6s already complete (%.0f MB)\n" "$cfg" "$((total / 1048576))"
    continue
  fi

  printf "%-6s %.0f of %.0f MB\n" "$cfg" "$((have / 1048576))" "$((total / 1048576))"
  for _ in $(seq 1 "$ATTEMPTS"); do
    have=$(stat -f%z "$out" 2>/dev/null || stat -c%s "$out" 2>/dev/null || echo 0)
    [ "$have" -ge "$total" ] && break
    curl -sL --max-time 420 -r "${have}-" -o "$out.chunk" "$url" 2>/dev/null
    got=$(stat -f%z "$out.chunk" 2>/dev/null || stat -c%s "$out.chunk" 2>/dev/null || echo 0)
    if [ "$got" -gt 0 ]; then
      cat "$out.chunk" >> "$out" && rm -f "$out.chunk"
      printf "  %-6s %.0f MB\n" "$cfg" "$(( $(stat -f%z "$out" 2>/dev/null || stat -c%s "$out") / 1048576 ))"
    fi
  done

  have=$(stat -f%z "$out" 2>/dev/null || stat -c%s "$out" 2>/dev/null || echo 0)
  if [ "$have" -ge "$total" ]; then
    echo "  $cfg complete"
  else
    echo "  $cfg INCOMPLETE — re-run to continue from $((have / 1048576)) MB" >&2
  fi
done
