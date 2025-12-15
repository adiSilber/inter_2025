#!/bin/bash
#SBATCH --job-name=download_model
#SBATCH --output=/specific/scratches/parallel/evyataroren-2025-12-31/MAFAT_HSRC/logs/slurm-download_model-%j.out
#SBATCH --error=/specific/scratches/parallel/evyataroren-2025-12-31/MAFAT_HSRC/logs/slurm-download_model-%j.err
#SBATCH --partition=cpu-killable
#SBATCH --account gpu-research 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --requeue
#SBATCH --signal=B:TERM@60
#SBATCH --chdir=/specific/scratches/parallel/evyataroren-2025-12-31/MAFAT_HSRC/
#SBATCH --open-mode=append
#SBATCH --export=ALL

# download with `sbatch download.slurm <path> <model1> [model2] [model3] ...`
# view with tail -F -n +1 slurm-<JOBID>.out
set -Eeuo pipefail

SAVE_DIR="${1:?Save directory required as first argument}"
shift  # Remove first argument, leaving all model names in $@
if [ $# -eq 0 ]; then
  echo "[$(date -Iseconds)] ERROR: At least one model name required"
  exit 1
fi

BASE="/specific/scratches/parallel/evyataroren-2025-12-31/MAFAT_HSRC"
CONDA_BIN="${BASE}/.miniconda3/envs/mafathsrc/bin"

# keep everything off $HOME
mkdir -p "${BASE}/${SAVE_DIR}" "${BASE}/.hf_home" "${BASE}/.hf_datasets" "${BASE}/.hf_transformers"

# auth + caches in scratch
TOKEN_FILE="${BASE}/.secrets/hf_read_token"
if [ ! -f "${TOKEN_FILE}" ]; then
  echo "[$(date -Iseconds)] ERROR: Token file not found: ${TOKEN_FILE}"
  exit 1
fi
if [ ! -r "${TOKEN_FILE}" ]; then
  echo "[$(date -Iseconds)] ERROR: Token file not readable: ${TOKEN_FILE}"
  echo "[$(date -Iseconds)] Attempting to fix permissions..."
  chmod 600 "${TOKEN_FILE}" 2>/dev/null || {
    echo "[$(date -Iseconds)] Failed to fix permissions. Please run: chmod 600 ${TOKEN_FILE}"
    exit 1
  }
fi
export HF_TOKEN="$(cat "${TOKEN_FILE}")"
if [ -z "${HF_TOKEN}" ]; then
  echo "[$(date -Iseconds)] ERROR: Token file is empty: ${TOKEN_FILE}"
  exit 1
fi
export HF_HOME="${BASE}/.hf_home"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${BASE}/.hf_datasets"
export TRANSFORMERS_CACHE="${BASE}/.hf_transformers"
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "[$(date -Iseconds)] node=${SLURM_JOB_NODELIST:-N/A} job=${SLURM_JOB_ID:-N/A}"

# preemption-friendly: exit 0 on TERM so Slurm can requeue cleanly
trap 'echo "[$(date -Iseconds)] TERM received; exiting cleanly for requeue"; exit 0' TERM

# ensure CLI exists (prefer the env-provided one)
if ! "${CONDA_BIN}/huggingface-cli" --version >/dev/null 2>&1; then
  "${CONDA_BIN}/pip" install --upgrade --no-cache-dir 'huggingface_hub[hf_transfer]'
fi

# download all models
for MODEL_NAME in "$@"; do
  echo "[$(date -Iseconds)] Processing model: ${MODEL_NAME}"
  # Extract just the model name (after the last /)
  MODEL_BASENAME="${MODEL_NAME##*/}"
  TARGET="${BASE}/${SAVE_DIR}/${MODEL_BASENAME}"
  
  # download (progress is shown by default)
  set -x
  "${CONDA_BIN}/hf" download \
    --token "${HF_TOKEN}" \
    --local-dir "${TARGET}" \
    "${MODEL_NAME}"
  set +x

  echo "[$(date -Iseconds)] Download complete for ${MODEL_NAME}."

  # Verify downloaded files
  echo "[$(date -Iseconds)] Verifying downloaded files for ${MODEL_NAME}..."
  "${CONDA_BIN}/python" ${BASE}/scripts_and_misc/verify_download.py --name "${MODEL_NAME}" --path "${TARGET}" --token "${HF_TOKEN}"
  VERIFY_EXIT_CODE=$?

  if [ $VERIFY_EXIT_CODE -eq 0 ]; then
    echo "[$(date -Iseconds)] Verification passed for ${MODEL_NAME}."
    echo "[$(date -Iseconds)] Removing cache folders for ${MODEL_NAME}..."
    rm -rf "${TARGET}/.cache"
    echo "[$(date -Iseconds)] Cache cleaned up for ${MODEL_NAME}."
  else
    echo "[$(date -Iseconds)] Verification failed for ${MODEL_NAME} with exit code ${VERIFY_EXIT_CODE}."
    exit $VERIFY_EXIT_CODE
  fi
done

echo "[$(date -Iseconds)] All models downloaded and verified successfully."