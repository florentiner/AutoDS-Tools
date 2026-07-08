#!/usr/bin/env bash
# Build the AutoDS-MLAB base image and one or more family images.
#
# Usage:
#   harbor/environments/build.sh                # base + tabular
#   harbor/environments/build.sh tabular nlp    # base + the listed families
#   harbor/environments/build.sh --all          # base + every family
#
# Images are tagged autods-mlab-base:latest and autods-mlab-<family>:latest.
# Task environment/Dockerfiles reference these via `FROM autods-mlab-<family>`.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

ALL_FAMILIES=(tabular tabred nlp vision graph clrs numeric)

if [[ "${1:-}" == "--all" ]]; then
  families=("${ALL_FAMILIES[@]}")
elif [[ "$#" -gt 0 ]]; then
  families=("$@")
else
  families=(tabular)
fi

echo ">> Building autods-mlab-base:latest"
docker build -f harbor/environments/base/Dockerfile -t autods-mlab-base:latest .

for fam in "${families[@]}"; do
  dockerfile="harbor/environments/${fam}/Dockerfile"
  if [[ ! -f "$dockerfile" ]]; then
    echo "!! Skipping ${fam}: ${dockerfile} not found" >&2
    continue
  fi
  echo ">> Building autods-mlab-${fam}:latest"
  docker build -f "$dockerfile" -t "autods-mlab-${fam}:latest" .
done

echo ">> Done. Images:"
docker images --format '{{.Repository}}:{{.Tag}}\t{{.Size}}' | grep '^autods-mlab-' || true
