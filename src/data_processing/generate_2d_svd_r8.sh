#!/bin/bash
# Convert every h5 in mrco_h5/{test,train} to a rank-8 2D-SVD reconstruction
# and write the outputs to mrco_h5_svd/{2d_svd_test,2d_svd_train}.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATOR="${SCRIPT_DIR}/svd_dataset_generator.py"

SRC_ROOT="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/mrco_h5"
DST_ROOT="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/mrco_h5_svd"
RANK=8
MODE=2d
PARALLEL=4  # number of concurrent files to process

for split in test train; do
    src_dir="${SRC_ROOT}/${split}"
    dst_dir="${DST_ROOT}/2d_svd_${split}"
    mkdir -p "${dst_dir}"

    echo "=== ${split}: ${src_dir} -> ${dst_dir} ==="
    shopt -s nullglob
    for f in "${src_dir}"/*.h5; do
        base="$(basename "${f}" .h5)"
        out="${dst_dir}/${base}_svd${MODE}_r${RANK}.h5"
        if [ -e "${out}" ]; then
            echo "  skip (already exists): ${out}"
            continue
        fi
        python3 "${GENERATOR}" \
            --input "${f}" \
            --output-dir "${dst_dir}" \
            --mode "${MODE}" \
            --rank "${RANK}" &
        while (( $(jobs -r | wc -l) >= PARALLEL )); do
            wait -n
        done
    done
    wait
    shopt -u nullglob
done

echo "Done."
