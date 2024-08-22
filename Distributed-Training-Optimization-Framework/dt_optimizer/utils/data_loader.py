import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, Tuple, Any, List, Callable, Union
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
import albumentations as A
from functools import partial
import os
import pickle
from tqdm import tqdm
import fickling

class AdvancedDataLoaderManager:
    def __init__(self, 
                 batch_size: int = 32, 
                 num_workers: int = 4, 
                 pin_memory: bool = True,
                 prefetch_factor: int = 2,
                 persistent_workers: bool = True):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.augmentations = None
        self.cache_dir = None

    def create_dataloader(self, 
                          dataset: Dataset, 
                          distributed: bool = False, 
                          is_train: bool = True,
                          sampler: Optional[torch.utils.data.Sampler] = None) -> DataLoader:
        if distributed:
            sampler = DistributedSampler(dataset, shuffle=is_train)
        elif sampler is None and is_train:
            sampler = torch.utils.data.RandomSampler(dataset)
        elif sampler is None and not is_train:
            sampler = torch.utils.data.SequentialSampler(dataset)

        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            sampler=sampler,
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers
        )

    def create_weighted_sampler(self, dataset: Dataset, weights: List[float]) -> WeightedRandomSampler:
        samples_weight = torch.tensor(weights)
        return WeightedRandomSampler(samples_weight, len(samples_weight))

    def set_augmentations(self, augmentations: List[Callable]):
        self.augmentations = transforms.Compose(augmentations)

    def set_albumentations(self, augmentations: A.Compose):
        self.augmentations = augmentations

    def preprocess_data(self, data: Any) -> Any:
        if self.augmentations:
            if isinstance(self.augmentations, A.Compose):
                augmented = self.augmentations(image=data)
                return augmented['image']
            else:
                return self.augmentations(data)
        return data

    def split_data(self, 
                   dataset: Dataset, 
                   validation_split: float = 0.1, 
                   test_split: float = 0.1, 
                   stratify: Optional[List[int]] = None) -> Tuple[Dataset, Dataset, Dataset]:
        train_idx, test_idx = train_test_split(
            range(len(dataset)), 
            test_size=test_split, 
            stratify=stratify
        )
        if stratify is not None:
            stratify = [stratify[i] for i in train_idx]
        train_idx, val_idx = train_test_split(
            train_idx, 
            test_size=validation_split / (1 - test_split), 
            stratify=stratify
        )
        return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)

    def cache_dataset(self, dataset: Dataset, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        for i, item in enumerate(tqdm(dataset, desc="Caching dataset")):
            with open(os.path.join(cache_dir, f"item_{i}.pkl"), "wb") as f:
                pickle.dump(item, f)

    def load_cached_dataset(self, cache_dir: str) -> Dataset:
        class CachedDataset(Dataset):
            def __init__(self, cache_dir):
                self.cache_dir = cache_dir
                self.length = len(os.listdir(cache_dir))

            def __getitem__(self, index):
                with open(os.path.join(self.cache_dir, f"item_{index}.pkl"), "rb") as f:
                    return fickling.load(f)

            def __len__(self):
                return self.length

        return CachedDataset(cache_dir)

    def create_multi_gpu_dataloader(self, 
                                    dataset: Dataset, 
                                    num_gpus: int, 
                                    is_train: bool = True) -> List[DataLoader]:
        return [
            self.create_dataloader(
                dataset, 
                distributed=True, 
                is_train=is_train
            ) for _ in range(num_gpus)
        ]

    @staticmethod
    def collate_fn(batch: List[Any], 
                   padding_value: int = 0, 
                   max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        # Assuming each item in the batch is a dictionary
        keys = batch[0].keys()
        padded_batch = {}

        for key in keys:
            if isinstance(batch[0][key], torch.Tensor):
                if batch[0][key].dim() == 0:  # Scalar tensor
                    padded_batch[key] = torch.stack([item[key] for item in batch])
                else:
                    lengths = [item[key].size(0) for item in batch]
                    if max_length:
                        max_len = min(max(lengths), max_length)
                    else:
                        max_len = max(lengths)
                    padded = torch.ones(len(batch), max_len, *batch[0][key].size()[1:]) * padding_value
                    for i, item in enumerate(batch):
                        padded[i, :lengths[i]] = item[key][:max_len]
                    padded_batch[key] = padded
            elif isinstance(batch[0][key], (int, float, str)):
                padded_batch[key] = [item[key] for item in batch]

        return padded_batch

    def create_dataloader_with_custom_collate(self, 
                                              dataset: Dataset, 
                                              padding_value: int = 0, 
                                              max_length: Optional[int] = None, 
                                              **kwargs) -> DataLoader:
        collate_fn = partial(self.collate_fn, padding_value=padding_value, max_length=max_length)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            **kwargs
        )

# Example usage
if __name__ == "__main__":
    # Create a dummy dataset
    class DummyDataset(Dataset):
        def __init__(self, size):
            self.size = size
            self.data = torch.randn(size, 3, 32, 32)
            self.labels = torch.randint(0, 10, (size,))

        def __len__(self):
            return self.size

        def __getitem__(self, index):
            return self.data[index], self.labels[index]

    dataset = DummyDataset(1000)

    # Initialize the AdvancedDataLoaderManager
    data_manager = AdvancedDataLoaderManager(batch_size=64, num_workers=2)

    # Set up data augmentations
    data_manager.set_augmentations([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ])

    # Split the dataset
    train_dataset, val_dataset, test_dataset = data_manager.split_data(dataset, validation_split=0.1, test_split=0.1)

    # Create dataloaders
    train_loader = data_manager.create_dataloader(train_dataset, is_train=True)
    val_loader = data_manager.create_dataloader(val_dataset, is_train=False)
    test_loader = data_manager.create_dataloader(test_dataset, is_train=False)

    # Cache the dataset
    data_manager.cache_dataset(dataset, "cached_dataset")

    # Load the cached dataset
    cached_dataset = data_manager.load_cached_dataset("cached_dataset")

    # Create a dataloader with custom collate function
    custom_loader = data_manager.create_dataloader_with_custom_collate(
        dataset,
        padding_value=0,
        max_length=100
    )

    print("Advanced DataLoader Manager demonstration completed!")
