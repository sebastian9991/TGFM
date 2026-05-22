#!/bin/bash
# Usage: ./submit_chain.sh [num_chunks] [extra args passed to job.sh]
# Example: ./submit_chain.sh 8

set -e


#Resolve the directory this script lives in.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/job.sh"

num_chunks=${1:-5}
shift || true  # Remove num_chunks from $@, pass rest to job.sh

echo "Submitting chain of $num_chunks jobs (3h each = $((num_chunks * 3))h total)"

# First job: no dependency
jobid=$(sbatch --parsable --time=03:00:00 $JOB_SCRIPT "$@")
echo "  Chunk 1/$num_chunks: $JOB_SCRIPT"

# Subsequent jobs: depend on the previous one succeeding
for i in $(seq 2 $num_chunks); do
    jobid=$(sbatch --parsable \
            --dependency=afterany:$jobid \
            --kill-on-invalid-dep=yes \
            --time=03:00:00 \
            $JOB_SCRIPT "$@")
    echo "  Chunk $i/$num_chunks: $jobid"
done

echo "Chain submitted."
