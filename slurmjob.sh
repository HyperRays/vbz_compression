#!/bin/bash
#SBATCH --job-name=vbz_throughput
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=logs/vbz_%A_%a.out
#SBATCH --error=logs/vbz_%A_%a.err
#SBATCH --array=0-6%3

mkdir -p logs

# ── Configuration: one entry per dataset ──────────────────────────
# Format: "label:fast5_dir"
DATASETS=(
"d1_sars-cov-2_r94:/mnt/galactica/skuvalekar/genome_data/test/data/d1_sars-cov-2_r94"
"d2_ecoli_r94:/mnt/galactica/skuvalekar/genome_data/test/data/d2_ecoli_r94"
"d3_yeast_r94:/mnt/galactica/skuvalekar/genome_data/test/data/d3_yeast_r94"
"d4_green_algae_r94:/mnt/galactica/skuvalekar/genome_data/test/data/d4_green_algae_r94"
"d5_human_na12878_r94:/mnt/galactica/skuvalekar/genome_data/test/data/d5_human_na12878_r94"
"d6_ecoli_r104:/mnt/galactica/skuvalekar/genome_data/test/data/d6_ecoli_r104"
"d7_saureus_r104:/mnt/galactica/skuvalekar/genome_data/test/data/d7_saureus_r104"
)

# ── Which dataset this task should process ────────────────────────
INDEX=$SLURM_ARRAY_TASK_ID   # 0‑based, like the bash array
if [ -z "${DATASETS[$INDEX]}" ]; then
    echo "Error: invalid array index $INDEX" >&2
    exit 1
fi

entry="${DATASETS[$INDEX]}"
LABEL="${entry%%:*}"
FAST5_DIR="${entry#*:}"

echo "Task $INDEX – Processing dataset: $LABEL"
echo "FAST5 directory: $FAST5_DIR"

# ── Generate sorted file list ─────────────────────────────────────
FILELIST="${LABEL}_files.txt"
find "$FAST5_DIR" -name "*.fast5" | sort > "$FILELIST"
echo "First 10 files in list:"
head -10 "$FILELIST"

# ── Run the throughput binary ─────────────────────────────────────
./build/bin/vbz_fast5_throughput \
    --fast5_list="$FILELIST" \
    --checkpoint="${LABEL}_run.ckpt" \
    --log="${LABEL}_run.log" \
    --log_interval=10

echo "Task $INDEX finished."