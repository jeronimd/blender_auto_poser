import msgpack
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from poser.augmentation import MirrorSkeleton, RandomRotation
from utils.constants import Constants
from utils.logger import Logger
from utils.path_manager import PathManager


class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.path_manager = PathManager()
        self.logger = Logger.setup(__name__, "DEBUG")

        self.data = []

        with open(self.path_manager.parsed_animations_file, 'rb') as f:
            animations = msgpack.unpack(f)

        for _, frames in animations.items():
            for frame_data in frames.values():
                self.data.append({
                    'locations': [frame_data[bone]['location'] for bone in Constants.BONE_IDX],
                    'rotations': [frame_data[bone]['rotation'] for bone in Constants.BONE_IDX]
                })

        self.logger.info(f"Total loaded frames: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame = self.data[idx]
        locations = torch.tensor(frame['locations'], dtype=torch.float32)
        rotations = torch.tensor(frame['rotations'], dtype=torch.float32)
        return {
            'locations': locations,
            'rotations': rotations,
        }


class BatchedDataset(Dataset):
    def __init__(self, base_dataset, batch_size, augment):
        self.logger = Logger.setup(__name__, "DEBUG")
        self.base_dataset = base_dataset
        self.batch_size = batch_size
        self.augment = augment
        self.mirror_aug = MirrorSkeleton()
        self.rotation_aug = RandomRotation()

        self.logger.info(f"Batched dataset: Frames={len(self.base_dataset)}, Batch size={self.batch_size}, Augmentation={self.augment}")

    def __len__(self):
        return (len(self.base_dataset) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.base_dataset))

        locations = []
        rotations = []
        for i in range(start_idx, end_idx):
            sample = self.base_dataset[i]
            locations.append(sample['locations'])
            rotations.append(sample['rotations'])

        # Stack into batched tensors
        locations = torch.stack(locations)
        rotations = torch.stack(rotations)

        if self.augment:
            locations, rotations = self.mirror_aug(locations, rotations)
            locations, rotations = self.rotation_aug(locations, rotations)

        return {'joint_positions': locations, 'joint_rotations': rotations}


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, persistent_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

        self.training_dataset: BatchedDataset
        self.validation_dataset: BatchedDataset

    def setup(self, stage=None):
        # Prepare Dataset
        dataset = BaseDataset()

        # Calculate actual lengths from dataset size
        dataset_size = len(dataset)
        train_size = int(dataset_size * 0.8)
        val_size = dataset_size - train_size

        # Split dataset into training and validation sets
        generator = torch.Generator().manual_seed(torch.initial_seed())
        training_dataset, validation_dataset = random_split(dataset, [train_size, val_size], generator=generator)

        # Create batched datasets
        self.training_dataset = BatchedDataset(training_dataset, self.batch_size, True)
        self.validation_dataset = BatchedDataset(validation_dataset, self.batch_size, False)

    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=1, shuffle=True, num_workers=self.num_workers, collate_fn=self._collate_fn, persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=self._collate_fn, persistent_workers=self.persistent_workers)

    @staticmethod
    def _collate_fn(batch):
        return batch[0]
