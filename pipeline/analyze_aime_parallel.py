#!/home/ADV_2526a/evyataroren/inter_2025/.miniconda3/envs/inter25_vllm/bin/python
#SBATCH --job-name=aime_parallel
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --array=0-9
#SBATCH --output=/home/ADV_2526a/evyataroren/inter_2025/logs/aime_parallel_%A_%a.out
#SBATCH --error=/home/ADV_2526a/evyataroren/inter_2025/logs/aime_parallel_%A_%a.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable

import os
import sys
import glob
import dill
from math import ceil

PROJECT_ROOT = "/home/ADV_2526a/evyataroren/inter_2025"
sys.path.insert(0, PROJECT_ROOT)

from experiments.adisi_things.attention_analysis_util import attention_to_question_per_token, detect_attention_spikes
from experiments.utils import printdp

EXPERIMENT_ID = "160652"
ARTIFACTS_DIR = "/home/ADV_2526a/evyataroren/inter_2025/artifacts_adisi/"
OUTPUT_DIR = "/home/ADV_2526a/evyataroren/inter_2025/artifacts_adisi/aime_analysis_results/"
NUM_JOBS = 10

JOB_ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(glob.glob(f"{ARTIFACTS_DIR}{EXPERIMENT_ID}*_datapoints__*.pkl"))

    # Split files across jobs
    files_per_job = ceil(len(files) / NUM_JOBS)
    start_idx = JOB_ID * files_per_job
    end_idx = min(start_idx + files_per_job, len(files))
    my_files = files[start_idx:end_idx]

    if not my_files:
        print(f"Job {JOB_ID}: No files to process")
        return

    print(f"Job {JOB_ID}: Processing {len(my_files)} files (indices {start_idx}-{end_idx-1})")

    results = []
    for filepath in my_files:
        print(f"\nLoading {os.path.basename(filepath)}")
        with open(filepath, 'rb') as f:
            datapoints = dill.load(f)
        if isinstance(datapoints, tuple):
            datapoints = datapoints[0]

        for i, dp in enumerate(datapoints):
            printdp(dp, i)
            attn = attention_to_question_per_token(dp)
            spikes = detect_attention_spikes(dp, threshold=0.18, consecutive=2)
            avg = sum(attn)/len(attn) if attn else 0.0
            print(f"avg_attention: {avg:.4f}, has_spike: {spikes['has_spike']}")
            results.append({'question_id': dp.question_id, 'attention': attn, 'avg': avg, 'spikes': spikes})

    with open(f"{OUTPUT_DIR}/results_{JOB_ID:03d}.pkl", 'wb') as f:
        dill.dump(results, f)
    print(f"\nJob {JOB_ID}: Saved {len(results)} results")

if __name__ == "__main__":
    main()
