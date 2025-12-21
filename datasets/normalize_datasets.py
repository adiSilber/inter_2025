"""
Dataset Normalization Script

This script normalizes multiple datasets into a unified JSON format.
Each dataset is converted to a standard format with question/answer pairs
and original attributes preserved.

To add a new dataset:
1. Create a new loader function following the pattern: load_<dataset_name>
2. Add the function to the DATASET_LOADERS dictionary
3. The loader should return a list of dicts with 'question', 'answer', and 'originalattr'
"""

import json
import csv
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


def load_aime(base_path: Path) -> List[Dict[str, Any]]:
    """Load AIME dataset from CSV."""
    csv_path = base_path / "AIME_1983_2024" / "AIME_Dataset_1983_2024.csv"
    
    result = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            result.append({
                "question": row.get("Question", ""),
                "answer": row.get("Answer", ""),
                "originalattr": {
                    "id": row.get("ID", ""),
                    "year": row.get("Year", ""),
                    "problem_number": row.get("Problem Number", ""),
                    "part": row.get("Part", "")
                }
            })
    
    return result


def load_math500(base_path: Path) -> List[Dict[str, Any]]:
    """Load MATH-500 dataset from JSONL."""
    jsonl_path = base_path / "MATH-500" / "test.jsonl"
    
    result = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            result.append({
                "question": data.get("problem", ""),
                "answer": data.get("answer", ""),
                "originalattr": {
                    "solution": data.get("solution", ""),
                    "subject": data.get("subject", ""),
                    "level": data.get("level", ""),
                    "unique_id": data.get("unique_id", "")
                }
            })
    
    return result


def load_arc_challenge(base_path: Path) -> List[Dict[str, Any]]:
    """Load ARC-Challenge dataset from parquet."""
    parquet_path = base_path / "ai2_arc" / "ARC-Challenge" / "test-00000-of-00001.parquet"
    
    df = pd.read_parquet(parquet_path)
    result = []
    
    for _, row in df.iterrows():
        # Format question with choices
        question = row['question']
        choices = row['choices']
        choices_dict = {}
        
        if isinstance(choices, dict) and 'text' in choices and 'label' in choices:
            # Convert numpy arrays to lists
            choices_dict = {
                'text': list(choices['text']) if hasattr(choices['text'], '__iter__') else choices['text'],
                'label': list(choices['label']) if hasattr(choices['label'], '__iter__') else choices['label']
            }
            question += "\nChoices:\n"
            for label, text in zip(choices_dict['label'], choices_dict['text']):
                question += f"{label}. {text}\n"
        
        result.append({
            "question": question.strip(),
            "answer": str(row.get("answerKey", "")),
            "originalattr": {
                "id": str(row.get("id", "")),
                "choices": choices_dict
            }
        })
    
    return result


def load_arc_easy(base_path: Path) -> List[Dict[str, Any]]:
    """Load ARC-Easy dataset from parquet."""
    parquet_path = base_path / "ai2_arc" / "ARC-Easy" / "test-00000-of-00001.parquet"
    
    df = pd.read_parquet(parquet_path)
    result = []
    
    for _, row in df.iterrows():
        # Format question with choices
        question = row['question']
        choices = row['choices']
        choices_dict = {}
        
        if isinstance(choices, dict) and 'text' in choices and 'label' in choices:
            # Convert numpy arrays to lists
            choices_dict = {
                'text': list(choices['text']) if hasattr(choices['text'], '__iter__') else choices['text'],
                'label': list(choices['label']) if hasattr(choices['label'], '__iter__') else choices['label']
            }
            question += "\nChoices:\n"
            for label, text in zip(choices_dict['label'], choices_dict['text']):
                question += f"{label}. {text}\n"
        
        result.append({
            "question": question.strip(),
            "answer": str(row.get("answerKey", "")),
            "originalattr": {
                "id": str(row.get("id", "")),
                "choices": choices_dict
            }
        })
    
    return result


def load_humaneval(base_path: Path) -> List[Dict[str, Any]]:
    """Load HumanEval dataset from parquet."""
    parquet_path = base_path / "openai_humaneval" / "openai_humaneval" / "test-00000-of-00001.parquet"
    
    df = pd.read_parquet(parquet_path)
    result = []
    
    for _, row in df.iterrows():
        result.append({
            "question": row.get("prompt", ""),
            "answer": row.get("canonical_solution", ""),
            "originalattr": {
                "task_id": row.get("task_id", ""),
                "test": row.get("test", ""),
                "entry_point": row.get("entry_point", "")
            }
        })
    
    return result


# Dictionary mapping dataset names to their loader functions
DATASET_LOADERS = {
    "AIME_1983_2024": load_aime,
    "MATH-500": load_math500,
    "ARC-Challenge": load_arc_challenge,
    "ARC-Easy": load_arc_easy,
    "openai_humaneval": load_humaneval,
}


def normalize_datasets(datasets_path: str, output_path: str, selected_datasets: List[str] = None):
    """
    Normalize multiple datasets into a unified JSON format.
    
    Args:
        datasets_path: Path to the datasets directory
        output_path: Path where the normalized JSON will be saved
        selected_datasets: List of dataset names to include (None = all)
    """
    base_path = Path(datasets_path)
    normalized_data = {}
    
    # Determine which datasets to process
    datasets_to_process = selected_datasets if selected_datasets else DATASET_LOADERS.keys()
    
    for dataset_name in datasets_to_process:
        if dataset_name not in DATASET_LOADERS:
            print(f"Warning: No loader found for dataset '{dataset_name}'. Skipping.")
            continue
        
        print(f"Processing {dataset_name}...")
        try:
            loader_func = DATASET_LOADERS[dataset_name]
            dataset_entries = loader_func(base_path)
            normalized_data[dataset_name] = dataset_entries
            print(f"  ✓ Loaded {len(dataset_entries)} entries from {dataset_name}")
        except Exception as e:
            print(f"  ✗ Error loading {dataset_name}: {str(e)}")
    
    # Save to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(normalized_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Normalized data saved to {output_path}")
    print(f"Total datasets: {len(normalized_data)}")
    print(f"Total entries: {sum(len(entries) for entries in normalized_data.values())}")


if __name__ == "__main__":
    import sys
    
    # Default paths relative to this script's location
    script_dir = Path(__file__).parent
    datasets_path = str(script_dir / "datasets")
    output_path = str(script_dir / "datasets" / "normalized_datasets.json")
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        datasets_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    normalize_datasets(datasets_path, output_path)
