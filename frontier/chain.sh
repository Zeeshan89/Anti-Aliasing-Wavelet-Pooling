#!/bin/bash
# Submit a manifest as chained pack8 jobs.
#   frontier/chain.sh <manifest> [repeats=3]
# Each block of 8 lines gets <repeats> jobs linked with afterany, so a run that
# needs more than one walltime resumes automatically. Finished runs exit
# immediately (done.flag), so extra repeats cost only a few minutes.
set -e
MANIFEST=$1; REPEATS=${2:-3}
DIR=$(dirname "$(realpath "$0")")
N=$(grep -cv '^\s*\(#\|$\)' "$MANIFEST" || true)
TOTAL=$(wc -l < "$MANIFEST")
for ((start = 1; start <= TOTAL; start += 8)); do
    prev=""
    for ((r = 0; r < REPEATS; r++)); do
        dep=${prev:+--dependency=afterany:$prev}
        prev=$(sbatch --parsable $dep "$DIR/pack8.sbatch" "$MANIFEST" "$start")
        echo "lines $start-$((start + 7)) repeat $r -> job $prev"
    done
done
echo "$N runs submitted from $MANIFEST"
