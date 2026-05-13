#!/bin/bash
# Usage: ./submit_chain.sh [num_chunks] [extra args passed to job.sh]
# Example: ./submit_chain.sh 8 --config-override foo=bar

set -e

num_chunks=${1:-5}
shift || true  # Remove num_chunks from $@, pass rest to job.sh

echo "Submitting chain of $num_chunks jobs (3h each = $((num_chunks * 3))h total)"

# First job: no dependency
jobid=$(sbatch --parsable --time=03:00:00 job.sh "$@")
echo "  Chunk 1/$num_chunks: $jobid"

# Subsequent jobs: depend on the previous one succeeding
for i in $(seq 2 $num_chunks); do
    jobid=$(sbatch --parsable \
            --dependency=afterok:$jobid \
            --kill-on-invalid-dep=yes \
            --time=03:00:00 \
            job.sh "$@")
    echo "  Chunk $i/$num_chunks: $jobid"
done

echo "Chain submitted."
