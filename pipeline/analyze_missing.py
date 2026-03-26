#!/home/ADV_2526a/evyataroren/inter_2025/.miniconda3/envs/inter25_vllm/bin/python
#SBATCH --job-name=aime_missing
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --output=/home/ADV_2526a/evyataroren/inter_2025/logs/aime_missing_%j.out
#SBATCH --error=/home/ADV_2526a/evyataroren/inter_2025/logs/aime_missing_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable

import os
import sys
import dill

PROJECT_ROOT = "/home/ADV_2526a/evyataroren/inter_2025"
sys.path.insert(0, PROJECT_ROOT)

from experiments.adisi_things.attention_analysis_util import attention_to_question_per_token, detect_attention_spikes
from experiments.utils import printdp

ARTIFACTS_DIR = "/home/ADV_2526a/evyataroren/inter_2025/artifacts_adisi/"
OUTPUT_DIR = "/home/ADV_2526a/evyataroren/inter_2025/artifacts_adisi/aime_analysis_results/"
EXPERIMENT_ID = "160652"

# Files from failed jobs (jobs 2, 4, 6, 8 that hit corrupted files)
# These files are valid but weren't saved because the jobs crashed after loading them
MISSING_FILES = ['19_21', '2_4', '36_38', '45_47', '51_53', '60_62', '68_70', '6_8']

# Known corrupted files - skip these
CORRUPTED = {'21_23', '38_40', '53_55', '70_72'}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []

    for file_idx in MISSING_FILES:
        if file_idx in CORRUPTED:
            print(f"Skipping corrupted file: {file_idx}")
            continue

        filepath = f"{ARTIFACTS_DIR}{EXPERIMENT_ID}_adisi_aime_near_miss_WITH_activations_74_final_answers_260305_v2_datapoints__{file_idx}.pkl"

        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue

        print(f"\n{'='*50}")
        print(f"Loading {file_idx}...")

        try:
            with open(filepath, 'rb') as f:
                datapoints = dill.load(f)
            if isinstance(datapoints, tuple):
                datapoints = datapoints[0]

            file_results = []
            for i, dp in enumerate(datapoints):
                printdp(dp, i)
                attn = attention_to_question_per_token(dp)
                spikes = detect_attention_spikes(dp, threshold=0.18, consecutive=2)
                avg = sum(attn)/len(attn) if attn else 0.0
                print(f"avg_attention: {avg:.4f}, has_spike: {spikes['has_spike']}")
                file_results.append({'question_id': dp.question_id, 'attention': attn, 'avg': avg, 'spikes': spikes})

            all_results.extend(file_results)

            # Save incrementally after each file
            output_path = f"{OUTPUT_DIR}/results_missing.pkl"
            with open(output_path, 'wb') as f:
                dill.dump(all_results, f)
            print(f"Saved {len(all_results)} total results so far")

        except Exception as e:
            print(f"Error processing {file_idx}: {e}")
            continue

    print(f"\n{'='*50}")
    print(f"DONE: Processed {len(all_results)} datapoints total")

if __name__ == "__main__":
    main()
