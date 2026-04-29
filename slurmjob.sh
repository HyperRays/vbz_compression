#!/bin/bash
#SBATCH --job-name=vbz_throughput
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/vbz_%j.out
#SBATCH --error=logs/vbz_%j.err

mkdir -p logs

# ── Configuration ────────────────────────────────────────────────────
# dataset_label:fast5_dir pairs
DATASETS=(
"d1_sars-cov-2_r94:/mnt/galactica/skuvalekar/genome_data/test/data/d1_sars-cov-2_r94"
"d2_ecoli_r94:/mnt/galactica/skuvalekar/genome_data/test/data/d2_ecoli_r94"
"d3_yeast_r94:/mnt/galactica/skuvalekar/genome_data/test/data/d3_yeast_r94"
"d4_green_algae_r94:/mnt/galactica/skuvalekar/genome_data/test/data/d4_green_algae_r94"
"d5_human_na12878_r94:/mnt/galactica/skuvalekar/genome_data/test/data/d5_human_na12878_r94"
"d6_ecoli_r104:/mnt/galactica/skuvalekar/genome_data/test/data/d6_ecoli_r104"
"d7_saureus_r104:/mnt/galactica/skuvalekar/genome_data/test/data/d7_saureus_r104"
)

for entry in "${DATASETS[@]}"; do
    LABEL="${entry%%:*}"
    FAST5_DIR="${entry#*:}"

    # Generate sorted file list
    find "$FAST5_DIR" -name "*.fast5" | sort > "${LABEL}_files.txt"
    echo "First 10 files for ${LABEL}:"
    head -10 "${LABEL}_files.txt" 

    ./build/bin/vbz_fast5_throughput \
        --fast5_list="${LABEL}_files.txt" \
        --checkpoint="${LABEL}_run.ckpt" \
        --log="${LABEL}_run.log" \
        --log_interval=10
done