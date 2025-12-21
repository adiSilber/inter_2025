#!/bin/bash
#SBATCH --job-name=download_hf
#SBATCH --output=logs/slurm-download_model-%j.out
#SBATCH --error=logs/slurm-download_model-%j.err
#SBATCH --partition=cpu-killable
#SBATCH --account=gpu-research
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --requeue
#SBATCH --signal=B:TERM@60
#SBATCH --open-mode=append
#SBATCH --export=ALL


set -euo pipefail


echo "=================================================="
echo "SLURM Job Information"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory (PWD): $PWD"
echo "Arguments: $@"
echo "=================================================="
echo ""

"utils/download_hf.sh" "$@"

exit_code=$?

echo ""
echo "=================================================="
echo "Job completed with exit code: $exit_code"
echo "=================================================="

exit $exit_code

