import os
import json
from typing import Any, Dict, Optional, Tuple, Union

import h5py

import numpy as np
import torch
from torch.utils.data import Dataset, Dataset, DataLoader
from lightning import LightningDataModule
from pytorch_lightning.utilities.combined_loader import CombinedLoader


def compute_velocities(positions: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Compute velocities using finite differences from positions.
    
    Parameters:
        positions (np.ndarray): Array of shape (T, N, D), where
                                T = number of timesteps,
                                N = number of particles,
                                D = spatial dimensions (e.g., 2 for 2D).
        dt (float): Time step size between position samples (default: 1.0).
    
    Returns:
        np.ndarray: Velocities of shape (T, N, D).
    """
    velocities = np.zeros_like(positions)
    
    # forward difference for the first timestep
    velocities[0] = (positions[1] - positions[0]) / dt

    # fentral difference for interior timesteps
    velocities[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)

    # backward difference for the last timestep
    velocities[-1] = (positions[-1] - positions[-2]) / dt

    return velocities


class CFDDataset(Dataset):
    def __init__(self, x, v):
        self.x = x
        self.v = v
        
    def __getitem__(self, idx):
        return self.x[idx], self.v[idx]
        
    def __len__(self):
        return len(self.x)


class CFDDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/CFD",
        dataset_name: str = "2DTGV",
        desired_num_timesteps: int = 5,
        desired_num_particles: int = 2000,
        split_ratios: Tuple[float, float, float] = (0.8, 0.0, 0.2),
        batch_size: int = 256,
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
        self.zeroframe = None

        self.train_dataloaders = []
        self.val_dataloaders = []
        self.test_dataloaders = []

        split = "train.h5"     
        path = os.path.join(data_dir, dataset_name, split)  
        with h5py.File(path, "r") as f:
            positions = f[f"00000/position"][:]     # first key (sample), shape: (126, 2500, 2)/(num_timesteps, num_particles, coordinates) for 2DTGV

        metadata_path = os.path.join(data_dir, dataset_name, "metadata.json")
        with open(metadata_path, "r") as metadata_file:
            metadata = json.load(metadata_file)

        # get velocities for each timestep
        dt = metadata["dt"] * metadata["write_every"]
        velocities = compute_velocities(positions, dt=dt)
            
        data_tensor = torch.tensor(positions, dtype=torch.float32)
        vel_tensor = torch.tensor(velocities, dtype=torch.float32)

        # subsample data (time steps and particles)

        # subsample timesteps
        timestep_indices = torch.linspace(0, data_tensor.shape[0] - 1, desired_num_timesteps, dtype=torch.long)
        data_tensor = data_tensor[timestep_indices]
        vel_tensor = vel_tensor[timestep_indices]

        # subsample particles
        particle_indices = torch.randperm(data_tensor.shape[1])[:desired_num_particles]
        data_tensor = data_tensor[:, particle_indices, :]
        vel_tensor = vel_tensor[:, particle_indices, :]

        self.num_timesteps = data_tensor.shape[0]

        generator=torch.Generator().manual_seed(self.seed)

        self.xs = []
        self.vs = []
        self.test_xs = []
        self.test_vs = []

        # indices for shuffling the data
        num_particles = data_tensor.shape[1]    # can be lower than desired_num_particles if not enough particles 
        shuffled_indices = torch.randperm(num_particles, generator=generator)

        # shuffle data (shuffle particles)
        data_tensor = data_tensor[:, shuffled_indices, :]
        vel_tensor = vel_tensor[:, shuffled_indices, :]

        # for splitting data
        split_index_1 = int(num_particles * split_ratios[0])  
        split_index_2 = int(num_particles * (split_ratios[0] + split_ratios[1]))
 
        for idx in range(self.num_timesteps): 
            print("Processing timestep: ", idx)

            frame_data_X = data_tensor[idx]
            frame_data_V = vel_tensor[idx]   

            # split data
            train_data_X = frame_data_X[:split_index_1]
            val_data_X = frame_data_X[split_index_1:split_index_2]
            test_data_X = frame_data_X[split_index_2:]

            train_data_V = frame_data_V[:split_index_1]
            val_data_V = frame_data_V[split_index_1:split_index_2]
            test_data_V = frame_data_V[split_index_2:]

            frame_dataset = CFDDataset(frame_data_X, frame_data_V)
            train_dataset = CFDDataset(train_data_X, train_data_V)
            val_dataset = CFDDataset(val_data_X, val_data_V)
            test_dataset = CFDDataset(test_data_X, test_data_V)

            self.xs.append(train_data_X)
            self.vs.append(train_data_V)

            self.train_dataloaders.append(
                DataLoader(
                    train_dataset,
                    batch_size=self.batch_size,
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    shuffle=True,
                    drop_last=True,
                )
            )
            self.val_dataloaders.append(
                DataLoader(
                    val_dataset,
                    batch_size=self.batch_size,
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    shuffle=False,
                    drop_last=True,
                )
            )

            self.test_dataloaders.append(
                DataLoader(
                    test_dataset,
                    batch_size=test_data_X.shape[0],   #  Potentially change this to what we had in MFM
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    shuffle=False,
                    drop_last=False,
                )
            )

    def train_dataloader(self):
        return CombinedLoader(self.train_dataloaders, mode="min_size")

    def val_dataloader(self):
        return CombinedLoader(self.val_dataloaders, mode="min_size")

    def test_dataloader(self):
        return CombinedLoader(self.test_dataloaders, "max_size")


if __name__ == "__main__":
    _ = CFDDataModule()
    print('Done!')
