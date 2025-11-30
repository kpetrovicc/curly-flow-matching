import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, TensorDataset, Dataset
import numpy as np
import scanpy as sc
import matplotlib

matplotlib.use("Agg")
import scvelo as scv

from src.models.components.utils import get_cell_velocities
from src import data, utils
from pytorch_lightning.utilities.combined_loader import CombinedLoader

class OceansDataset(Dataset):
    def __init__(self, x, v):
        self.x = x
        self.v = v
        
    def __getitem__(self, idx):
        return self.x[idx], self.v[idx]
        
    def __len__(self):
        return len(self.x)

class OceansDataModule(LightningDataModule):
    def __init__(
        self,
        adata_path: Optional[str] = None,
        data_dir: str = "data/",
        split_ratios: Union[
            int, Tuple[int, int], Tuple[float, float]
        ] = 1,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: Optional[int] = 42,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.batch_size = batch_size

        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.seed = seed

        self.train_dataloaders = []
        self.val_dataloaders = []
        self.test_dataloaders = []

        if adata_path is None:
            adata_path = "/slurm-storage/katpet/InteractingFlows/data/oceans/laz_oceans_preproc_gom_vortex_UNCOUPLED_SMALL.npz"

        loaded_data = np.load(adata_path)
        positions, velocities = loaded_data["positions"], loaded_data["velocities"]

        num_times = positions.shape[0]
        num_particles = positions.shape[1]
        d = positions.shape[-1]
        self.num_timesteps = num_times

        generator=torch.Generator().manual_seed(self.seed)

        self.xs = []
        self.vs = []
        self.test_xs = []
        self.test_vs = []

        train_ts = [0, 2, 4, 6, 8]
        test_ts = [1, 3, 5, 7]

        for t in range(num_times):
            x_ts = torch.tensor(positions[t])
            v_ts = torch.tensor(velocities[t])
            split_index = int(len(x_ts) * split_ratios[0])
            shuffled_indices = torch.randperm(num_particles, generator=generator)
            x_t = x_ts[shuffled_indices]
            v_t = v_ts[shuffled_indices]

            if t in train_ts:
                train_data = OceansDataset(x_t[:split_index], v_t[:split_index])
                val_data = OceansDataset(x_t[split_index:], v_t[split_index:])
                self.xs.append(x_t[:split_index])
                self.vs.append(v_t[:split_index])
                self.train_dataloaders.append(DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory))
                self.val_dataloaders.append(DataLoader(val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory))
            else:
                self.test_xs.append(x_t)
                self.test_vs.append(v_t)
                test_data = OceansDataset(x_t, v_t)
                self.test_dataloaders.append(DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory))

    def train_dataloader(self):
        return CombinedLoader(self.train_dataloaders, mode="min_size")

    def val_dataloader(self):
        return CombinedLoader(self.val_dataloaders, mode="min_size")

    def test_dataloader(self):
        return CombinedLoader(self.test_dataloaders, "max_size")

if __name__ == "__main__":
    _ = OceansDataModule()