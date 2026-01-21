import os
import random
import json
import pandas as pd
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class question_item:
    q : str
    a : str
    question_id : str
    misc_specific_misc : Dict[str, Any]


class dataset_loader:
    def __init__(self,  base_path: str, seed: int = -1) -> None:
        """
        when seed is -1, no randomization is done. (shuffling the dataset).
        base_path: path to the dataset folder. class will look in `base_path/dataset_name/` for dataset files.
        """
        self.base_path = base_path
        self.seed = seed
        self.data = []
        self.index = 0
        self.loaded = False

    def load_data(self) -> None:
        if self.loaded:
            return
        if self.seed != -1:
            prev = random.getstate()
            random.seed(self.seed)
            self._load_data()
            random.shuffle(self.data)
            random.setstate(prev)
        self.loaded = True

    def _load_data(self):
        raise NotImplementedError("load_data method must be implemented by subclasses")

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self, k:int=1) -> List[question_item]:
        """
        returns next k samples from the dataset as a list of question_item objects
        """
        if self.index >= len(self.data):
            raise StopIteration
        
        batch = self.data[self.index : self.index + k]
        self.index += k
        
        if not batch:
            raise StopIteration

        return batch

    def __len__(self) -> int:
        return len(self.data)

class AI2ARCLoader(dataset_loader):
    def _load_data(self):
        # Try ARC-Challenge first
        path = os.path.join(self.base_path, "ai2_arc", "ARC-Challenge", "test-00000-of-00001.parquet")
        if not os.path.exists(path):
             path = os.path.join(self.base_path, "ai2_arc", "ARC-Easy", "test-00000-of-00001.parquet")
        
        if os.path.exists(path):
            df = pd.read_parquet(path)
            for _, row in df.iterrows():
                q = row['question']
                choices = row['choices']
                # choices is dict {'text': [], 'label': []}
                labels = choices['label']
                texts = choices['text']
                formatted_choices = "\n".join([f"{l}. {t}" for l, t in zip(labels, texts)])
                full_q = f"{q}\n{formatted_choices}"
                
                self.data.append(question_item(
                    q=full_q,
                    a=row['answerKey'],
                    question_id="aiarc"+str(row['id']),
                    misc_specific_misc={'id': row['id']}
                ))

class AIMELoader(dataset_loader):
    def _load_data(self):
        path = os.path.join(self.base_path, "AIME_1983_2024", "AIME_Dataset_1983_2024.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                self.data.append(question_item(
                    q=row['Question'],
                    a=str(row['Answer']),
                    question_id=str(row['ID']),
                    misc_specific_misc={
                        'ID': "aim"+row['ID'],
                        'Year': row['Year'],
                        'Problem Number': row['Problem Number']
                    }
                ))

class MATH500Loader(dataset_loader):
    def _load_data(self):
        path = os.path.join(self.base_path, "MATH-500", "test.jsonl")
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    self.data.append(question_item(
                        q=item['problem'],
                        a=item['solution'],
                        question_id="math500"+item['unique_id'],
                        misc_specific_misc={
                            'answer': item['answer'],
                            'subject': item['subject'],
                            'level': item['level'],
                            'unique_id': item['unique_id']
                        }
                    ))

class HumanEvalLoader(dataset_loader):
    def _load_data(self):
        path = os.path.join(self.base_path, "openai_humaneval", "openai_humaneval", "test-00000-of-00001.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            for _, row in df.iterrows():
                self.data.append(question_item(
                    q=row['prompt'],
                    a=row['canonical_solution'],
                    question_id="humaneval"+row['task_id'],
                    misc_specific_misc={
                        'task_id': row['task_id'],
                        'test': row['test'],
                        'entry_point': row['entry_point']
                    }
                ))

class GPQALoader(dataset_loader):
    def _load_data(self):
        try:
            from datasets import load_dataset
            # Try to use local cache if it exists
            cache_dir = os.path.join(self.base_path, "gpqa", ".cache")
            if not os.path.exists(cache_dir):
                cache_dir = None 
            
            # Load gpqa_diamond
            ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", cache_dir=cache_dir)
            for idx, row in enumerate(ds):
                self.data.append(question_item(
                    q=row['Question'],
                    a=row['Correct Answer'],
                    question_id="gpqa"+str(row['Record ID']),
                    misc_specific_misc={
                        'Record ID': row['Record ID'],
                        'Incorrect Answer 1': row['Incorrect Answer 1'],
                        'Incorrect Answer 2': row['Incorrect Answer 2'],
                        'Incorrect Answer 3': row['Incorrect Answer 3']
                    }
                ))
        except Exception as e:
            print(f"Failed to load GPQA: {e}")

class SimpleDatasetLoader(dataset_loader):
    def _load_data(self):
        path = os.path.join(self.base_path, "simple", "simple.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                content = json.load(f)
                for idx, item in enumerate(content.get('data', [])):
                    self.data.append(question_item(
                        q=item['question'],
                        a=item['answer'],
                        question_id=f"simple_{idx}",
                        misc_specific_misc={}
                    ))

class aggregate_shuffle_strategy:
    SEQUENTIAL = 0
    RANDOM = 1
    ROUND_ROBIN = 2
    
class aggregated_dataset_loader:
    def __init__(self, datasets: list[dataset_loader], seed = -1, strategy : aggregate_shuffle_strategy = aggregate_shuffle_strategy.SEQUENTIAL, base_path=None) -> None:
        """
        dataset_loaders: list of dataset_loader objects
        """
        self.seed = seed
        self.strategy = strategy
        self.loaders = []
        if base_path is None:
            base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "datasets")

        # Instantiate the loader classes
        for ds in datasets:
            if issubclass(ds, dataset_loader):
                loader = ds(base_path, seed)
                loader._load_data()
                self.loaders.append(loader)
            elif isinstance(ds, dataset_loader):
                ds._load_data()
                self.loaders.append(ds)
            else:
                raise ValueError("datasets must be a list of dataset_loader classes or instances")
        
        if strategy == aggregate_shuffle_strategy.RANDOM:
            pass
        elif strategy == aggregate_shuffle_strategy.ROUND_ROBIN:
            self.rr_index = 0
        else:
            self.current_loader_index = 0

    def __iter__(self):
        return self

    def __next__(self, k:int=1) -> List[question_item]:
        """
        returns next k samples from the aggregated dataset loaders as a list of question_item objects
        """
        if self.strategy == aggregate_shuffle_strategy.SEQUENTIAL:
            if self.current_loader_index >= len(self.loaders):
                raise StopIteration
            try:
                sample = self.loaders[self.current_loader_index].__next__(k)
                return sample
            except StopIteration:
                self.current_loader_index += 1
                return self.__next__(k)
        elif self.strategy == aggregate_shuffle_strategy.ROUND_ROBIN:
            if len(self.loaders) == 0:
                raise StopIteration
            current_loader = self.loaders[self.rr_index % len(self.loaders)]
            try:
                sample = current_loader.__next__(k)
                self.rr_index += 1
                return sample
            except StopIteration:
                self.loaders.pop(self.rr_index % len(self.loaders))
                return self.__next__(k)
        elif self.strategy == aggregate_shuffle_strategy.RANDOM:
            if len(self.loaders) == 0:
                raise StopIteration
            if self.seed != -1:
                prev = random.getstate()
                random.seed(self.seed)
            current_loader = random.choice(self.loaders)
            if self.seed != -1:
                random.setstate(prev)
            try:
                sample = current_loader.__next__(k)
                return sample
            except StopIteration:
                self.loaders.remove(current_loader)
                return self.__next__(k)

    def __len__(self) -> int:
        total = 0
        for loader in self.loaders:
            total += len(loader.data)
        return total
