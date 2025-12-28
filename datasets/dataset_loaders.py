from networkx import union
import os
import json
import pandas as pd
import random
import glob

class dataset_loaders:
    def __init__(self,  base_path:str, seed = -1) -> None:
        """
        when seed is -1, no randomization is done. (shuffling the dataset).
        base_path: path to the dataset folder. class will look in `base_path/dataset_name/` for dataset files.
        """
        self.base_path = base_path
        self.seed = seed
        self.data = []
        self.cursor = 0
        
    def __iter__(self):
        self.cursor = 0
        return self
        
    def __next__(self, k:int=1):
        """
        returns next k samples from the dataset.
        If k=1, returns a dict {'q': "", 'a': "", 'misc_specific_misc': {...}}
        If k>1, returns a list of such dicts.
        """
        if self.cursor >= len(self.data):
            raise StopIteration
            
        end = min(self.cursor + k, len(self.data))
        batch = self.data[self.cursor : end]
        self.cursor = end
        
        if k == 1 and len(batch) == 1:
            return batch[0]
        return batch

class AIME_Loader(dataset_loaders):
    def __init__(self, base_path: str, seed=-1) -> None:
        super().__init__(base_path, seed)
        self.dataset_name = "AIME_1983_2024"
        self._load_data()
        
    def _load_data(self):
        path = os.path.join(self.base_path, self.dataset_name, "AIME_Dataset_1983_2024.csv")
        if not os.path.exists(path):
            print(f"Warning: {path} not found.")
            return
            
        df = pd.read_csv(path)
        if self.seed != -1:
            df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
            
        for _, row in df.iterrows():
            self.data.append({
                'q': str(row['Question']),
                'a': str(row['Answer']),
                'misc_specific_misc': {
                    'ID': row['ID'],
                    'Year': row['Year'],
                    'Problem Number': row['Problem Number'],
                    'Part': row['Part']
                }
            })

class MATH500_Loader(dataset_loaders):
    def __init__(self, base_path: str, seed=-1) -> None:
        super().__init__(base_path, seed)
        self.dataset_name = "MATH-500"
        self._load_data()

    def _load_data(self):
        path = os.path.join(self.base_path, self.dataset_name, "test.jsonl")
        if not os.path.exists(path):
            print(f"Warning: {path} not found.")
            return

        with open(path, 'r') as f:
            lines = f.readlines()
        
        if self.seed != -1:
            random.seed(self.seed)
            random.shuffle(lines)
            
        for line in lines:
            item = json.loads(line)
            self.data.append({
                'q': item['problem'],
                'a': item['solution'],
                'misc_specific_misc': {
                    'answer': item['answer'],
                    'subject': item['subject'],
                    'level': item['level'],
                    'unique_id': item['unique_id']
                }
            })

class GPQA_Loader(dataset_loaders):
    def __init__(self, base_path: str, seed=-1) -> None:
        super().__init__(base_path, seed)
        self.dataset_name = "gpqa"
        self._load_data()

    def _load_data(self):
        target_dir = os.path.join(self.base_path, self.dataset_name)
        csv_files = glob.glob(os.path.join(target_dir, "**/*.csv"), recursive=True)
        
        if not csv_files:
             # Try loading from datasets library
             try:
                 from datasets import load_dataset
                 # Check if cache exists
                 cache_dir = os.path.join(target_dir, ".cache")
                 if not os.path.exists(cache_dir):
                     cache_dir = None
                 
                 ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", cache_dir=cache_dir)
                 for row in ds:
                    question = row['Question']
                    correct = row['Correct Answer']
                    incorrects = [row['Incorrect Answer 1'], row['Incorrect Answer 2'], row['Incorrect Answer 3']]
                    
                    choices = [correct] + incorrects
                    current_choices = choices.copy()
                    if self.seed != -1:
                        random.seed(self.seed)
                        random.shuffle(current_choices)
                    else:
                        random.shuffle(current_choices)

                    correct_idx = current_choices.index(correct)
                    answer_key = chr(65 + correct_idx)
                    
                    formatted_choices = "\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(current_choices)])
                    formatted_q = f"{question}\n{formatted_choices}"
                    
                    self.data.append({
                        'q': formatted_q,
                        'a': answer_key,
                        'misc_specific_misc': {
                            'correct_answer_text': correct,
                            'choices': current_choices
                        }
                    })
                 return
             except Exception as e:
                 print(f"Warning: No CSV files found for GPQA in {target_dir} and failed to load from datasets: {e}")
                 return

        all_rows = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                for _, row in df.iterrows():
                    all_rows.append(row)
            except Exception as e:
                print(f"Error reading {file}: {e}")
                
        if self.seed != -1:
            random.seed(self.seed)
            random.shuffle(all_rows)
            
        for row in all_rows:
            if 'Question' not in row: continue
            
            question = row['Question']
            correct = row['Correct Answer']
            incorrects = [row[c] for c in row.index if 'Incorrect Answer' in c and pd.notna(row[c])]
            
            choices = [correct] + incorrects
            current_choices = choices.copy()
            random.shuffle(current_choices)
            
            correct_idx = current_choices.index(correct)
            answer_key = chr(65 + correct_idx)
            
            formatted_choices = "\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(current_choices)])
            formatted_q = f"{question}\n{formatted_choices}"
            
            self.data.append({
                'q': formatted_q,
                'a': answer_key,
                'misc_specific_misc': {
                    'correct_answer_text': correct,
                    'choices': current_choices
                }
            })

class AI2ARC_Loader(dataset_loaders):
    def __init__(self, base_path: str, seed=-1, split='Challenge') -> None:
        super().__init__(base_path, seed)
        self.dataset_name = "ai2_arc"
        self.split = split
        self._load_data()

    def _load_data(self):
        target_dir = os.path.join(self.base_path, self.dataset_name, f"ARC-{self.split}")
        parquet_files = glob.glob(os.path.join(target_dir, "**/*.parquet"), recursive=True)
        
        if not parquet_files:
             print(f"Warning: No parquet files found for ARC-{self.split} in {target_dir}")
             return

        dfs = []
        for f in parquet_files:
            try:
                dfs.append(pd.read_parquet(f))
            except Exception as e:
                print(f"Error reading {f}: {e}")
        
        if not dfs:
            return

        df = pd.concat(dfs, ignore_index=True)
        
        if self.seed != -1:
            df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
            
        for _, row in df.iterrows():
            q_text = row['question']
            choices = row['choices'] 
            
            formatted_choices = []
            labels = choices.get('label', [])
            texts = choices.get('text', [])
            
            if hasattr(labels, 'tolist'): labels = labels.tolist()
            if hasattr(texts, 'tolist'): texts = texts.tolist()

            for label, text in zip(labels, texts):
                formatted_choices.append(f"({label}) {text}")
            
            formatted_q = f"{q_text}\n" + "\n".join(formatted_choices)
            
            self.data.append({
                'q': formatted_q,
                'a': row['answerKey'],
                'misc_specific_misc': {
                    'id': row['id'] if 'id' in row else None
                }
            })

class HumanEval_Loader(dataset_loaders):
    def __init__(self, base_path: str, seed=-1) -> None:
        super().__init__(base_path, seed)
        self.dataset_name = "openai_humaneval"
        self._load_data()

    def _load_data(self):
        target_dir = os.path.join(self.base_path, self.dataset_name, "openai_humaneval")
        parquet_files = glob.glob(os.path.join(target_dir, "**/*.parquet"), recursive=True)
        
        if not parquet_files:
             print(f"Warning: No parquet files found for HumanEval in {target_dir}")
             return

        dfs = []
        for f in parquet_files:
            try:
                dfs.append(pd.read_parquet(f))
            except Exception as e:
                print(f"Error reading {f}: {e}")
        
        if not dfs:
            return

        df = pd.concat(dfs, ignore_index=True)
        
        if self.seed != -1:
            df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
            
        for _, row in df.iterrows():
            self.data.append({
                'q': row['prompt'],
                'a': row['canonical_solution'],
                'misc_specific_misc': {
                    'task_id': row['task_id'],
                    'test': row['test'],
                    'entry_point': row['entry_point']
                }
            })

class aggregate_shuffle_strategy:
    SEQUENTIAL = 0
    RANDOM = 1
    EPOCH_RANDOM = 2
    
class aggregated_dataset_loader:
    def __init__(self, datasets: list[dataset_loaders], seed = -1, strategy : aggregate_shuffle_strategy = aggregate_shuffle_strategy.SEQUENTIAL) -> None:
        """
        dataset_loaders: list of dataset_loader objects
        """
        self.datasets = datasets
        self.seed = seed
        self.strategy = strategy
        self.all_data = []
        
        for ds in datasets:
            if hasattr(ds, 'data'):
                self.all_data.extend(ds.data)
        
        if strategy == aggregate_shuffle_strategy.RANDOM and seed != -1:
            random.seed(seed)
            random.shuffle(self.all_data)
            
        self.cursor = 0

    def __iter__(self):
        self.cursor = 0
        return self

    def __next__(self, k:int=1):
        """
        returns next k samples from the aggregated dataset loaders
        """
        if self.cursor >= len(self.all_data):
            raise StopIteration
            
        end = min(self.cursor + k, len(self.all_data))
        batch = self.all_data[self.cursor : end]
        self.cursor = end
        
        if k == 1 and len(batch) == 1:
            return batch[0]
        return batch
