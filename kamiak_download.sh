#!/bin/bash
#
# Download files from high performance cluster
#
. kamiak_config.sh

# Note both have trailing slashes
from="$remotessh:$remotedir"
to="$localdir"

# Logs, models, images
rsync -Pahuv \
    --include="results_*.txt" --include="*.pickle" \
    --include="slurm_logs/" --include="slurm_logs/*" \
    --include="$logFolder*/" --include="$logFolder*/*" --include="$logFolder*/*/*" \
    --include="$modelFolder*/" --include="$modelFolder*/*" --include="$modelFolder*/*/*" \
    --exclude="*" "$from" "$to"
