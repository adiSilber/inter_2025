"""
Prompts loader module for loading prompts from a folder with various selection methods.
"""
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple


class PromptsLoader:
    """
    A loader class for loading prompts from one or more directories with various selection strategies.
    """
    
    def __init__(self, prompts_folders: Union[str, Path, List[Union[str, Path]]]):
        """
        Initialize the prompts loader with one or more folder paths.
        
        Args:
            prompts_folders: Path or list of paths to folders containing prompt files
        """
        # Convert to list if single folder provided
        if not isinstance(prompts_folders, list):
            prompts_folders = [prompts_folders]
        
        self.prompts_folders = [Path(folder) for folder in prompts_folders]
        
        # Validate all folders
        for folder in self.prompts_folders:
            if not folder.exists():
                raise ValueError(f"Prompts folder does not exist: {folder}")
            if not folder.is_dir():
                raise ValueError(f"Path is not a directory: {folder}")
        
        self._cache_prompt_files()
    
    def _cache_prompt_files(self):
        """Cache all available prompt files from each folder."""
        # Store files per folder to maintain folder structure
        self.prompt_files_per_folder = []
        for folder in self.prompts_folders:
            folder_files = []
            for file_path in sorted(folder.iterdir()):
                if file_path.is_file():
                    folder_files.append(file_path)
            self.prompt_files_per_folder.append(folder_files)
        
        # Also maintain a flat list for backwards compatibility
        self.prompt_files = [file for folder_files in self.prompt_files_per_folder 
                            for file in folder_files]
    
    def _read_prompt(self, file_path: Path) -> str:
        """Read and return the content of a prompt file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def load_k_random(self, k: int, seed: Optional[int] = None) -> Union[Dict[str, str], Tuple[Dict[str, str], ...]]:
        """
        Load K random prompts across all folders.
        For multiple folders, randomly samples from the combined pool and returns separate subsets per folder.
        
        Args:
            k: Total number of prompts to load across all folders
            seed: Random seed for reproducibility (optional)
            
        Returns:
            If single folder: Dictionary mapping prompt names to their content
            If multiple folders: Tuple of dictionaries, one per folder
        """
        if k > len(self.prompt_files):
            raise ValueError(f"Cannot load {k} prompts, only {len(self.prompt_files)} available")
        
        # Set seed if provided
        if seed is not None:
            random.seed(seed)
        
        # Randomly sample from all files across all folders
        selected_files = random.sample(self.prompt_files, k)
        
        # Single folder case - return single dict
        if len(self.prompts_folders) == 1:
            return {file.stem: self._read_prompt(file) for file in selected_files}
        
        # Multiple folders case - separate by folder
        results = []
        for folder, folder_files in zip(self.prompts_folders, self.prompt_files_per_folder):
            folder_selected = [f for f in selected_files if f in folder_files]
            results.append({file.stem: self._read_prompt(file) for file in folder_selected})
        
        return tuple(results)
    
    def load_k_top(self, k: int) -> Union[Dict[str, str], Tuple[Dict[str, str], ...]]:
        """
        Load the first K prompts evenly distributed across folders.
        For multiple folders, loads k/n prompts from each folder.
        
        Args:
            k: Total number of prompts to load
            
        Returns:
            If single folder: Dictionary mapping prompt names to their content
            If multiple folders: Tuple of dictionaries, one per folder
        """
        if k > len(self.prompt_files):
            raise ValueError(f"Cannot load {k} prompts, only {len(self.prompt_files)} available")
        
        # Single folder case
        if len(self.prompts_folders) == 1:
            selected_files = self.prompt_files[:k]
            return {file.stem: self._read_prompt(file) for file in selected_files}
        
        # Multiple folders case - distribute evenly
        n_folders = len(self.prompts_folders)
        k_per_folder = k // n_folders
        remainder = k % n_folders
        
        results = []
        for i, folder_files in enumerate(self.prompt_files_per_folder):
            # Add one extra prompt to first 'remainder' folders
            k_this_folder = k_per_folder + (1 if i < remainder else 0)
            
            if k_this_folder > len(folder_files):
                raise ValueError(f"Cannot load {k_this_folder} prompts from folder {self.prompts_folders[i]}, "
                               f"only {len(folder_files)} available")
            
            selected_files = folder_files[:k_this_folder]
            results.append({file.stem: self._read_prompt(file) for file in selected_files})
        
        return tuple(results)
    
    def load_by_index(self, indices: Union[List[int], List[List[int]]]) -> Union[Dict[str, str], Tuple[Dict[str, str], ...]]:
        """
        Load prompts by their indices in the sorted file list.
        For multiple folders, expects an array of arrays (one per folder).
        
        Args:
            indices: List of indices (for single folder) or list of lists (one per folder)
            
        Returns:
            If single folder: Dictionary mapping prompt names to their content
            If multiple folders: Tuple of dictionaries, one per folder
        """
        # Single folder case
        if len(self.prompts_folders) == 1:
            if not indices or (indices and isinstance(indices[0], list)):
                raise ValueError("For single folder, provide a flat list of indices")
            
            result = {}
            for idx in indices:
                if idx < 0 or idx >= len(self.prompt_files):
                    raise ValueError(f"Index {idx} out of range [0, {len(self.prompt_files)})")
                file = self.prompt_files[idx]
                result[file.stem] = self._read_prompt(file)
            
            return result
        
        # Multiple folders case - expect array of arrays
        if not isinstance(indices[0], list):
            raise ValueError("For multiple folders, provide a list of lists (one per folder)")
        
        if len(indices) != len(self.prompts_folders):
            raise ValueError(f"Expected {len(self.prompts_folders)} index arrays, got {len(indices)}")
        
        results = []
        for folder_idx, (folder_files, folder_indices) in enumerate(zip(self.prompt_files_per_folder, indices)):
            folder_result = {}
            for idx in folder_indices:
                if idx < 0 or idx >= len(folder_files):
                    raise ValueError(f"Index {idx} out of range [0, {len(folder_files)}) for folder {self.prompts_folders[folder_idx]}")
                file = folder_files[idx]
                folder_result[file.stem] = self._read_prompt(file)
            results.append(folder_result)
        
        return tuple(results)
    
    def load_by_name(self, names: Union[List[str], List[List[str]]]) -> Union[Dict[str, str], Tuple[Dict[str, str], ...]]:
        """
        Load prompts by their filenames (without extension).
        For multiple folders, expects an array of arrays (one per folder).
        
        Args:
            names: List of prompt names (for single folder) or list of lists (one per folder)
            
        Returns:
            If single folder: Dictionary mapping prompt names to their content
            If multiple folders: Tuple of dictionaries, one per folder
        """
        # Single folder case
        if len(self.prompts_folders) == 1:
            if not names or (names and isinstance(names[0], list)):
                raise ValueError("For single folder, provide a flat list of names")
            
            name_to_file = {file.stem: file for file in self.prompt_files}
            
            result = {}
            for name in names:
                if name not in name_to_file:
                    raise ValueError(f"Prompt '{name}' not found in {self.prompts_folders[0]}")
                result[name] = self._read_prompt(name_to_file[name])
            
            return result
        
        # Multiple folders case - expect array of arrays
        if not isinstance(names[0], list):
            raise ValueError("For multiple folders, provide a list of lists (one per folder)")
        
        if len(names) != len(self.prompts_folders):
            raise ValueError(f"Expected {len(self.prompts_folders)} name arrays, got {len(names)}")
        
        results = []
        for folder_idx, (folder_files, folder_names) in enumerate(zip(self.prompt_files_per_folder, names)):
            name_to_file = {file.stem: file for file in folder_files}
            
            folder_result = {}
            for name in folder_names:
                if name not in name_to_file:
                    raise ValueError(f"Prompt '{name}' not found in {self.prompts_folders[folder_idx]}")
                folder_result[name] = self._read_prompt(name_to_file[name])
            results.append(folder_result)
        
        return tuple(results)
    
    def get_available_prompts(self) -> Union[List[str], Tuple[List[str], ...]]:
        """
        Get a list of all available prompt names.
        
        Returns:
            If single folder: List of prompt names
            If multiple folders: Tuple of lists, one per folder
        """
        if len(self.prompts_folders) == 1:
            return [file.stem for file in self.prompt_files]
        
        return tuple([file.stem for file in folder_files] 
                    for folder_files in self.prompt_files_per_folder)
    
    def get_prompts_count(self) -> Union[int, Tuple[int, ...]]:
        """
        Get the total number of available prompts.
        
        Returns:
            If single folder: Number of prompts
            If multiple folders: Tuple of counts, one per folder
        """
        if len(self.prompts_folders) == 1:
            return len(self.prompt_files)
        
        return tuple(len(folder_files) for folder_files in self.prompt_files_per_folder)


def load_prompts(
    prompts_folders: Union[str, Path, List[Union[str, Path]]],
    method: str = "all",
    k: Optional[int] = None,
    indices: Optional[Union[List[int], List[List[int]]]] = None,
    names: Optional[Union[List[str], List[List[str]]]] = None,
    seed: Optional[int] = None
) -> Union[Dict[str, str], Tuple[Dict[str, str], ...]]:
    """
    Convenience function to load prompts with a single call.
    
    Args:
        prompts_folders: Path or list of paths to folders containing prompt files
        method: Loading method - "k_random", "k_top", "by_index", "by_name", or "all"
        k: Number of prompts for k_random or k_top methods
        indices: List of indices (single folder) or list of lists (multiple folders)
        names: List of names (single folder) or list of lists (multiple folders)
        seed: Random seed for k_random method
        
    Returns:
        If single folder: Dictionary mapping prompt names to their content
        If multiple folders: Tuple of dictionaries, one per folder
        
    Examples:
        # Single folder - Load 5 random prompts
        prompts = load_prompts("./prompts", method="k_random", k=5, seed=42)
        
        # Multiple folders - Load 6 random prompts across folders
        prompts = load_prompts(["./prompts1", "./prompts2"], method="k_random", k=6, seed=42)
        # Returns tuple: (dict1, dict2) with prompts distributed randomly
        
        # Multiple folders - Load top 6 prompts (3 from each folder)
        prompts = load_prompts(["./prompts1", "./prompts2"], method="k_top", k=6)
        # Returns tuple: (dict1, dict2) with 3 prompts each
        
        # Multiple folders - Load by indices
        prompts = load_prompts(["./prompts1", "./prompts2"], method="by_index", 
                              indices=[[0, 1], [2, 3]])
        # Returns tuple: (dict1, dict2)
        
        # Multiple folders - Load by names
        prompts = load_prompts(["./prompts1", "./prompts2"], method="by_name",
                              names=[["prompt1", "prompt2"], ["prompt3"]])
        # Returns tuple: (dict1, dict2)
    """
    loader = PromptsLoader(prompts_folders)
    
    if method == "k_random":
        if k is None:
            raise ValueError("Parameter 'k' is required for k_random method")
        return loader.load_k_random(k, seed)
    
    elif method == "k_top":
        if k is None:
            raise ValueError("Parameter 'k' is required for k_top method")
        return loader.load_k_top(k)
    
    elif method == "by_index":
        if indices is None:
            raise ValueError("Parameter 'indices' is required for by_index method")
        return loader.load_by_index(indices)
    
    elif method == "by_name":
        if names is None:
            raise ValueError("Parameter 'names' is required for by_name method")
        return loader.load_by_name(names)
    
    elif method == "all":
        counts = loader.get_prompts_count()
        if isinstance(counts, tuple):
            # Multiple folders
            all_indices = [list(range(count)) for count in counts]
            return loader.load_by_index(all_indices)
        else:
            # Single folder
            return loader.load_by_index(list(range(counts)))
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'k_random', 'k_top', 'by_index', 'by_name', or 'all'")
"=" * 60)
        print("SINGLE FOLDER EXAMPLES")
        print("=" * 60)
        print(f"Loading prompts from: {example_folder}")
        print(f"Available prompts: {len(list(example_folder.glob('*')))}")
        
        # Example 1: Load 3 random prompts
        print("\n1. Loading 3 random prompts:")
        prompts = load_prompts(example_folder, method="k_random", k=3, seed=42)
        for name, content in prompts.items():
            print(f"  - {name}: {len(content)} characters")
        
        # Example 2: Load top 2 prompts
        print("\n2. Loading top 2 prompts:")
        prompts = load_prompts(example_folder, method="k_top", k=2)
        for name, content in prompts.items():
            print(f"  - {name}: {len(content)} characters")
        
        # Example 3: Load by index
        print("\n3. Loading by indices [0, 2]:")
        prompts = load_prompts(example_folder, method="by_index", indices=[0, 2])
        for name, content in prompts.items():
            print(f"  - {name}: {len(content)} characters")
    
    # Multiple folders example
    folder1 = Path(__file__).parent / "play" / "injections" / "uninformative"
    folder2 = Path(__file__).parent / "play" / "question_formats"
    
    if folder1.exists() and folder2.exists():
        print("\n" + "=" * 60)
        print("MULTIPLE FOLDERS EXAMPLES")
        print("=" * 60)
        print(f"Folder 1: {folder1}")
        print(f"Folder 2: {folder2}")
        
        # Example 4: Load 6 random prompts across both folders
        print("\n4. Loading 6 random prompts across both folders:")
        prompts_tuple = load_prompts([folder1, folder2], method="k_random", k=6, seed=42)
        for i, prompts in enumerate(prompts_tuple, 1):
            print(f"  Folder {i}: {len(prompts)} prompts")
            for name in prompts.keys():
                print(f"    - {name}")
        
        # Example 5: Load top 4 prompts (2 from each folder)
        print("\n5. Loading top 4 prompts (2 per folder):")
        prompts_tuple = load_prompts([folder1, folder2], method="k_top", k=4)
        for i, prompts in enumerate(prompts_tuple, 1):
            print(f"  Folder {i}: {len(prompts)} prompts")
            for name in prompts.keys():
                print(f"    - {name}")
        
        # Example 6: Load by indices from multiple folders
        print("\n6. Loading by indices [[0, 1], [0]]:")
        prompts_tuple = load_prompts([folder1, folder2], method="by_index", 
                                     indices=[[0, 1], [0]])
        for i, prompts in enumerate(prompts_tuple, 1):
            print(f"  Folder {i}: {len(prompts)} prompts")
            for name in prompts.keys():
                print(f"    - {name}
            print(f"  - {name}: {len(content)} characters")
        
        # Example 3: Load by index
        print("\n3. Loading by indices [0, 2]:")
        prompts = load_prompts(example_folder, method="by_index", indices=[0, 2])
        for name, content in prompts.items():
            print(f"  - {name}: {len(content)} characters")
