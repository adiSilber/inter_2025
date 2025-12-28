"""
Dataset loader module for loading datasets from JSON files with various selection methods.
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any


class DatasetLoader:
    """
    A loader class for loading datasets from one or more JSON files with various selection strategies.
    """
    
    def __init__(self, dataset_files: Union[str, Path, List[Union[str, Path]]]):
        """
        Initialize the dataset loader with one or more JSON file paths.
        
        Args:
            dataset_files: Path or list of paths to JSON files containing datasets
        """
        # Convert to list if single file provided
        if not isinstance(dataset_files, list):
            dataset_files = [dataset_files]
        
        self.dataset_files = [Path(file) for file in dataset_files]
        
        # Validate all files
        for file in self.dataset_files:
            if not file.exists():
                raise ValueError(f"Dataset file does not exist: {file}")
            if not file.is_file():
                raise ValueError(f"Path is not a file: {file}")
        
        self._load_datasets()
    
    def _load_datasets(self):
        """Load all datasets from JSON files."""
        self.datasets_per_file = []
        self.dataset_names_per_file = []
        
        for file in self.dataset_files:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Store the entire dataset dict for this file
            self.datasets_per_file.append(data)
            # Store the dataset names (keys)
            self.dataset_names_per_file.append(sorted(data.keys()))
        
        # Flat list of all dataset names for backwards compatibility
        self.all_dataset_names = [name for names in self.dataset_names_per_file for name in names]
    
    def _filter_originalattr(
        self, 
        items: List[Dict[str, Any]], 
        include_originalattr: Union[bool, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter items based on originalattr inclusion settings.
        
        Args:
            items: List of dataset items
            include_originalattr: If False, removes originalattr entirely.
                                 If True, keeps all originalattr fields.
                                 If dict, keeps only specified fields from originalattr.
        
        Returns:
            Filtered list of items
        """
        if include_originalattr is True:
            return items
        
        result = []
        for item in items:
            filtered_item = {}
            for key, value in item.items():
                if key == "originalattr":
                    if include_originalattr is False:
                        # Skip originalattr entirely
                        continue
                    elif isinstance(include_originalattr, dict):
                        # Include only specified fields
                        filtered_attr = {k: v for k, v in value.items() 
                                       if k in include_originalattr}
                        if filtered_attr:
                            filtered_item[key] = filtered_attr
                else:
                    filtered_item[key] = value
            result.append(filtered_item)
        
        return result
    
    def load_k_random(
        self, 
        k: int, 
        seed: Optional[int] = None,
        include_originalattr: Union[bool, Dict[str, Any]] = True
    ) -> Union[Dict[str, List[Dict[str, Any]]], Tuple[Dict[str, List[Dict[str, Any]]], ...]]:
        """
        Load K random dataset items across all datasets in all files.
        For multiple files, randomly samples from the combined pool and returns separate subsets per file.
        
        Args:
            k: Total number of items to load across all datasets
            seed: Random seed for reproducibility (optional)
            include_originalattr: If False, excludes originalattr. If True, includes all.
                                 If dict, includes only specified originalattr fields.
            
        Returns:
            If single file: Dictionary mapping dataset names to lists of items
            If multiple files: Tuple of dictionaries, one per file
        """
        # Collect all items with their source info
        all_items = []
        for file_idx, (dataset_dict, names) in enumerate(zip(self.datasets_per_file, self.dataset_names_per_file)):
            for dataset_name in names:
                items = dataset_dict[dataset_name]
                for item_idx, item in enumerate(items):
                    all_items.append({
                        'file_idx': file_idx,
                        'dataset_name': dataset_name,
                        'item': item
                    })
        
        if k > len(all_items):
            raise ValueError(f"Cannot load {k} items, only {len(all_items)} available across all datasets")
        
        # Set seed if provided
        if seed is not None:
            random.seed(seed)
        
        # Randomly sample
        selected = random.sample(all_items, k)
        
        # Single file case
        if len(self.dataset_files) == 1:
            result = {}
            for item_info in selected:
                dataset_name = item_info['dataset_name']
                if dataset_name not in result:
                    result[dataset_name] = []
                result[dataset_name].append(item_info['item'])
            
            # Apply originalattr filtering
            for dataset_name in result:
                result[dataset_name] = self._filter_originalattr(result[dataset_name], include_originalattr)
            
            return result
        
        # Multiple files case
        results = []
        for file_idx in range(len(self.dataset_files)):
            file_result = {}
            for item_info in selected:
                if item_info['file_idx'] == file_idx:
                    dataset_name = item_info['dataset_name']
                    if dataset_name not in file_result:
                        file_result[dataset_name] = []
                    file_result[dataset_name].append(item_info['item'])
            
            # Apply originalattr filtering
            for dataset_name in file_result:
                file_result[dataset_name] = self._filter_originalattr(file_result[dataset_name], include_originalattr)
            
            results.append(file_result)
        
        return tuple(results)
    
    def load_k_top(
        self, 
        k: int,
        include_originalattr: Union[bool, Dict[str, Any]] = True
    ) -> Union[Dict[str, List[Dict[str, Any]]], Tuple[Dict[str, List[Dict[str, Any]]], ...]]:
        """
        Load the first K items evenly distributed across files.
        For multiple files, loads k/n items from each file.
        
        Args:
            k: Total number of items to load
            include_originalattr: If False, excludes originalattr. If True, includes all.
                                 If dict, includes only specified originalattr fields.
            
        Returns:
            If single file: Dictionary mapping dataset names to lists of items
            If multiple files: Tuple of dictionaries, one per file
        """
        # Count total items
        total_items = sum(sum(len(dataset_dict[name]) for name in names)
                         for dataset_dict, names in zip(self.datasets_per_file, self.dataset_names_per_file))
        
        if k > total_items:
            raise ValueError(f"Cannot load {k} items, only {total_items} available")
        
        # Single file case
        if len(self.dataset_files) == 1:
            result = {}
            items_collected = 0
            
            for dataset_name in self.dataset_names_per_file[0]:
                if items_collected >= k:
                    break
                
                items = self.datasets_per_file[0][dataset_name]
                items_to_take = min(len(items), k - items_collected)
                result[dataset_name] = items[:items_to_take]
                items_collected += items_to_take
            
            # Apply originalattr filtering
            for dataset_name in result:
                result[dataset_name] = self._filter_originalattr(result[dataset_name], include_originalattr)
            
            return result
        
        # Multiple files case - distribute evenly
        n_files = len(self.dataset_files)
        k_per_file = k // n_files
        remainder = k % n_files
        
        results = []
        for file_idx, (dataset_dict, names) in enumerate(zip(self.datasets_per_file, self.dataset_names_per_file)):
            k_this_file = k_per_file + (1 if file_idx < remainder else 0)
            
            file_result = {}
            items_collected = 0
            
            for dataset_name in names:
                if items_collected >= k_this_file:
                    break
                
                items = dataset_dict[dataset_name]
                items_to_take = min(len(items), k_this_file - items_collected)
                file_result[dataset_name] = items[:items_to_take]
                items_collected += items_to_take
            
            # Apply originalattr filtering
            for dataset_name in file_result:
                file_result[dataset_name] = self._filter_originalattr(file_result[dataset_name], include_originalattr)
            
            results.append(file_result)
        
        return tuple(results)
    
    def load_by_index(
        self, 
        indices: Union[Dict[str, List[int]], List[Dict[str, List[int]]]],
        include_originalattr: Union[bool, Dict[str, Any]] = True
    ) -> Union[Dict[str, List[Dict[str, Any]]], Tuple[Dict[str, List[Dict[str, Any]]], ...]]:
        """
        Load items by their indices within each dataset.
        For single file: expects dict mapping dataset names to lists of indices.
        For multiple files: expects list of dicts (one per file).
        
        Args:
            indices: Dict of {dataset_name: [indices]} for single file,
                    or list of such dicts for multiple files
            include_originalattr: If False, excludes originalattr. If True, includes all.
                                 If dict, includes only specified originalattr fields.
            
        Returns:
            If single file: Dictionary mapping dataset names to lists of items
            If multiple files: Tuple of dictionaries, one per file
        """
        # Single file case
        if len(self.dataset_files) == 1:
            if isinstance(indices, list):
                raise ValueError("For single file, provide a dict mapping dataset names to indices")
            
            result = {}
            for dataset_name, item_indices in indices.items():
                if dataset_name not in self.datasets_per_file[0]:
                    raise ValueError(f"Dataset '{dataset_name}' not found in file {self.dataset_files[0]}")
                
                items = self.datasets_per_file[0][dataset_name]
                result[dataset_name] = []
                
                for idx in item_indices:
                    if idx < 0 or idx >= len(items):
                        raise ValueError(f"Index {idx} out of range [0, {len(items)}) for dataset '{dataset_name}'")
                    result[dataset_name].append(items[idx])
            
            # Apply originalattr filtering
            for dataset_name in result:
                result[dataset_name] = self._filter_originalattr(result[dataset_name], include_originalattr)
            
            return result
        
        # Multiple files case
        if not isinstance(indices, list):
            raise ValueError("For multiple files, provide a list of dicts (one per file)")
        
        if len(indices) != len(self.dataset_files):
            raise ValueError(f"Expected {len(self.dataset_files)} index dicts, got {len(indices)}")
        
        results = []
        for file_idx, (dataset_dict, file_indices) in enumerate(zip(self.datasets_per_file, indices)):
            file_result = {}
            
            for dataset_name, item_indices in file_indices.items():
                if dataset_name not in dataset_dict:
                    raise ValueError(f"Dataset '{dataset_name}' not found in file {self.dataset_files[file_idx]}")
                
                items = dataset_dict[dataset_name]
                file_result[dataset_name] = []
                
                for idx in item_indices:
                    if idx < 0 or idx >= len(items):
                        raise ValueError(f"Index {idx} out of range [0, {len(items)}) for dataset '{dataset_name}'")
                    file_result[dataset_name].append(items[idx])
            
            # Apply originalattr filtering
            for dataset_name in file_result:
                file_result[dataset_name] = self._filter_originalattr(file_result[dataset_name], include_originalattr)
            
            results.append(file_result)
        
        return tuple(results)
    
    def load_by_name(
        self, 
        names: Union[List[str], List[List[str]]],
        include_originalattr: Union[bool, Dict[str, Any]] = True
    ) -> Union[Dict[str, List[Dict[str, Any]]], Tuple[Dict[str, List[Dict[str, Any]]], ...]]:
        """
        Load entire datasets by their names.
        For single file: expects list of dataset names.
        For multiple files: expects list of lists (one per file).
        
        Args:
            names: List of dataset names for single file,
                  or list of lists for multiple files
            include_originalattr: If False, excludes originalattr. If True, includes all.
                                 If dict, includes only specified originalattr fields.
            
        Returns:
            If single file: Dictionary mapping dataset names to lists of items
            If multiple files: Tuple of dictionaries, one per file
        """
        # Single file case
        if len(self.dataset_files) == 1:
            if names and isinstance(names[0], list):
                raise ValueError("For single file, provide a flat list of dataset names")
            
            result = {}
            for dataset_name in names:
                if dataset_name not in self.datasets_per_file[0]:
                    raise ValueError(f"Dataset '{dataset_name}' not found in file {self.dataset_files[0]}")
                result[dataset_name] = self.datasets_per_file[0][dataset_name].copy()
            
            # Apply originalattr filtering
            for dataset_name in result:
                result[dataset_name] = self._filter_originalattr(result[dataset_name], include_originalattr)
            
            return result
        
        # Multiple files case
        if not names or not isinstance(names[0], list):
            raise ValueError("For multiple files, provide a list of lists (one per file)")
        
        if len(names) != len(self.dataset_files):
            raise ValueError(f"Expected {len(self.dataset_files)} name lists, got {len(names)}")
        
        results = []
        for file_idx, (dataset_dict, file_names) in enumerate(zip(self.datasets_per_file, names)):
            file_result = {}
            
            for dataset_name in file_names:
                if dataset_name not in dataset_dict:
                    raise ValueError(f"Dataset '{dataset_name}' not found in file {self.dataset_files[file_idx]}")
                file_result[dataset_name] = dataset_dict[dataset_name].copy()
            
            # Apply originalattr filtering
            for dataset_name in file_result:
                file_result[dataset_name] = self._filter_originalattr(file_result[dataset_name], include_originalattr)
            
            results.append(file_result)
        
        return tuple(results)
    
    def get_available_datasets(self) -> Union[List[str], Tuple[List[str], ...]]:
        """
        Get a list of all available dataset names.
        
        Returns:
            If single file: List of dataset names
            If multiple files: Tuple of lists, one per file
        """
        if len(self.dataset_files) == 1:
            return self.dataset_names_per_file[0]
        
        return tuple(self.dataset_names_per_file)
    
    def get_dataset_counts(self) -> Union[Dict[str, int], Tuple[Dict[str, int], ...]]:
        """
        Get the number of items in each dataset.
        
        Returns:
            If single file: Dict mapping dataset names to item counts
            If multiple files: Tuple of dicts, one per file
        """
        if len(self.dataset_files) == 1:
            return {name: len(self.datasets_per_file[0][name]) 
                   for name in self.dataset_names_per_file[0]}
        
        return tuple({name: len(dataset_dict[name]) for name in names}
                    for dataset_dict, names in zip(self.datasets_per_file, self.dataset_names_per_file))


def load_datasets(
    dataset_files: Union[str, Path, List[Union[str, Path]]],
    method: str = "all",
    k: Optional[int] = None,
    indices: Optional[Union[Dict[str, List[int]], List[Dict[str, List[int]]]]] = None,
    names: Optional[Union[List[str], List[List[str]]]] = None,
    seed: Optional[int] = None,
    include_originalattr: Union[bool, Dict[str, Any]] = True
) -> Union[Dict[str, List[Dict[str, Any]]], Tuple[Dict[str, List[Dict[str, Any]]], ...]]:
    """
    Convenience function to load datasets with a single call.
    
    Args:
        dataset_files: Path or list of paths to JSON files containing datasets
        method: Loading method - "k_random", "k_top", "by_index", "by_name", or "all"
        k: Number of items for k_random or k_top methods
        indices: Dict of {dataset: [indices]} for single file, or list of dicts for multiple files
        names: List of dataset names for single file, or list of lists for multiple files
        seed: Random seed for k_random method
        include_originalattr: If False, excludes originalattr field.
                             If True, includes all originalattr fields.
                             If dict (e.g., {"year": True, "id": True}), includes only specified fields.
        
    Returns:
        If single file: Dictionary mapping dataset names to lists of items
        If multiple files: Tuple of dictionaries, one per file
        
    Examples:
        # Single file - Load 10 random items
        data = load_datasets("normalized_datasets.json", method="k_random", k=10, seed=42)
        
        # Single file - Load without originalattr
        data = load_datasets("normalized_datasets.json", method="k_top", k=5, 
                           include_originalattr=False)
        
        # Single file - Load with only specific originalattr fields
        data = load_datasets("normalized_datasets.json", method="k_top", k=5,
                           include_originalattr={"year": True, "id": True})
        
        # Multiple files - Load 20 random items across files
        data = load_datasets(["file1.json", "file2.json"], method="k_random", k=20, seed=42)
        
        # Single file - Load by indices
        data = load_datasets("normalized_datasets.json", method="by_index",
                           indices={"AIME_1983_2024": [0, 1, 2], "gpqa": [0]})
        
        # Multiple files - Load by indices
        data = load_datasets(["file1.json", "file2.json"], method="by_index",
                           indices=[{"AIME_1983_2024": [0, 1]}, {"gpqa": [0, 1, 2]}])
        
        # Single file - Load specific datasets by name
        data = load_datasets("normalized_datasets.json", method="by_name",
                           names=["AIME_1983_2024", "gpqa"])
    """
    loader = DatasetLoader(dataset_files)
    
    if method == "k_random":
        if k is None:
            raise ValueError("Parameter 'k' is required for k_random method")
        return loader.load_k_random(k, seed, include_originalattr)
    
    elif method == "k_top":
        if k is None:
            raise ValueError("Parameter 'k' is required for k_top method")
        return loader.load_k_top(k, include_originalattr)
    
    elif method == "by_index":
        if indices is None:
            raise ValueError("Parameter 'indices' is required for by_index method")
        return loader.load_by_index(indices, include_originalattr)
    
    elif method == "by_name":
        if names is None:
            raise ValueError("Parameter 'names' is required for by_name method")
        return loader.load_by_name(names, include_originalattr)
    
    elif method == "all":
        all_names = loader.get_available_datasets()
        if isinstance(all_names, tuple):
            # Multiple files
            return loader.load_by_name(list(all_names), include_originalattr)
        else:
            # Single file
            return loader.load_by_name(all_names, include_originalattr)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'k_random', 'k_top', 'by_index', 'by_name', or 'all'")


if __name__ == "__main__":
    # Example usage
    dataset_file = Path(__file__).parent / "datasets" / "normalized_datasets.json"
    
    if dataset_file.exists():
        print("=" * 60)
        print("SINGLE FILE EXAMPLES")
        print("=" * 60)
        print(f"Loading from: {dataset_file}")
        
        loader = DatasetLoader(dataset_file)
        available = loader.get_available_datasets()
        counts = loader.get_dataset_counts()
        
        print(f"\nAvailable datasets: {len(available)}")
        for name in available[:3]:  # Show first 3
            print(f"  - {name}: {counts[name]} items")
        
        # Example 1: Load 5 random items
        print("\n1. Loading 5 random items:")
        data = load_datasets(dataset_file, method="k_random", k=5, seed=42)
        for dataset_name, items in data.items():
            print(f"  {dataset_name}: {len(items)} items")
        
        # Example 2: Load 3 items without originalattr
        print("\n2. Loading 3 items without originalattr:")
        data = load_datasets(dataset_file, method="k_top", k=3, include_originalattr=False)
        for dataset_name, items in data.items():
            print(f"  {dataset_name}: {len(items)} items")
            if items:
                print(f"    Keys: {list(items[0].keys())}")
        
        # Example 3: Load with specific originalattr fields
        print("\n3. Loading with only 'year' and 'id' from originalattr:")
        data = load_datasets(dataset_file, method="k_top", k=3, 
                           include_originalattr={"year": True, "id": True})
        for dataset_name, items in data.items():
            print(f"  {dataset_name}: {len(items)} items")
            if items and "originalattr" in items[0]:
                print(f"    originalattr keys: {list(items[0]['originalattr'].keys())}")
        
        # Example 4: Load specific dataset by name
        if "AIME_1983_2024" in available:
            print("\n4. Loading AIME_1983_2024 dataset by name (first 2 items):")
            data = load_datasets(dataset_file, method="by_index",
                               indices={"AIME_1983_2024": [0, 1]})
            for dataset_name, items in data.items():
                print(f"  {dataset_name}: {len(items)} items")
                if items:
                    print(f"    First question: {items[0]['question'][:80]}...")
