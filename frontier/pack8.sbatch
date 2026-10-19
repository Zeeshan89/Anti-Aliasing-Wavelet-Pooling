#!/bin/bash -l
#SBATCH --account=cli138
#SBATCH --job-name=aa_pack8
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=01:55:00
#SBATCH -o out/pack8-%j.txt
#SBATCH -e out/pack8-%j.txt
#SBATCH -C nvme
#
# Runs up to 8 manifest lines (one per GCD) in parallel.
#   sbatch frontier/pack8.sbatch <manifest> <start line (1-based)>
# Each manifest line = train_ul.py arguments without --data/--device, e.g.
#   --model yolov8s --dataset dota --variant gaussian --seed 0
# Runs resume from last.pt and skip on done.flag, so resubmitting the same
# block (frontier/chain.sh) continues unfinished runs.
unset SLURM_EXPORT_ENV
MANIFEST=$(realpath "$1"); START=${2:-1}
source /lustre/orion/proj-shared/cli138/z7a/Project/frontier/common.sh
cd "$PROJECT_DIR"; mkdir -p out "$MIOPEN_ROOT"

# Stop trainings 6 min before walltime so last.pt is not cut mid-write.
left=$(squeue -h -j "$SLURM_JOB_ID" -o %L)
IFS=: read -r -a t <<< "${left#*-}"
secs=$(( ${#t[@]} == 3 ? 10#${t[0]}*3600 + 10#${t[1]}*60 + 10#${t[2]} : 10#${t[0]}*60 + 10#${t[1]} ))
budget=$(( secs - 360 ))
echo "job $SLURM_JOB_ID  manifest=$MANIFEST start=$START  budget=${budget}s"

pids=(); lines=()
for i in $(seq 0 7); do
    LINE=$(sed -n "$((START + i))p" "$MANIFEST")
    [[ -z "$LINE" || "$LINE" == \#* ]] && continue
    ds=$(sed -n 's/.*--dataset[ =]\([^ ]*\).*/\1/p' <<< "$LINE")
    data=$(stage_dataset "$ds")     # sequential loop, so each tar is extracted once
    mkdir -p "$MIOPEN_ROOT/gcd$i"
    tag=$(tr ' ' '\n' <<< "$LINE" | grep -v '^--' | paste -sd_)
    log="out/${tag}_${SLURM_JOB_ID}_gcd$i.log"
    echo "  [gcd$i] $LINE  (data: $data, log: $log)"
    ROCR_VISIBLE_DEVICES=$i HIP_VISIBLE_DEVICES=$i \
    MIOPEN_USER_DB_PATH="$MIOPEN_ROOT/gcd$i" MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_ROOT/gcd$i" \
        timeout --signal=INT --kill-after=120 "$budget" \
        python -u train_ul.py $LINE --data "$data" --device 0 > "$log" 2>&1 &
    pids+=($!); lines+=("$LINE")
done

fail=0
for k in "${!pids[@]}"; do
    wait "${pids[$k]}"; rc=$?
    case $rc in
        0)   echo "  [ok]      ${lines[$k]}" ;;
        124|130) echo "  [walltime] ${lines[$k]} (will resume)" ;;
        *)   echo "  [FAIL rc=$rc] ${lines[$k]}" >&2; fail=1 ;;
    esac
done
exit $fail
