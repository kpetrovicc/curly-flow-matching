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

from src.models.components.single_marginal_utils import get_cell_velocities
from src import utils


class PairedSamplingDataset(Dataset):
    """Wrapper dataset that returns pairs of samples."""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, _):
        # Sample two random indices
        idx1 = torch.randint(0, len(self.base_dataset), (1,)).item()
        idx2 = torch.randint(0, len(self.base_dataset), (1,)).item()

        # Get UMAP coordinates for both samples
        x0 = self.base_dataset[idx1][0]  # [0] to get UMAP coordinates
        x1 = self.base_dataset[idx2][0]

        v0 = self.base_dataset[idx1][1]  # [1] to get velocities
        v1 = self.base_dataset[idx2][1]

        return x0, x1, v0, v1


class DeepCycleDataModule(LightningDataModule):
    def __init__(
        self,
        adata_path: Optional[str] = None,
        data_dir: str = "data/",
        train_val_test_split: Union[
            int, Tuple[int, int, int], Tuple[float, float, float]
        ] = 1,
        batch_size: int = 64,
        hvgs: Optional[int] = 10,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.dim = hvgs
        self.seed = seed

        if adata_path is None:
            adata_path = "/slurm-storage/katpet/InteractingFlows/data/DeepCycle/fibroblast_velocity.h5ad"
        adata = sc.read_h5ad(adata_path)

        if self.hparams.hvgs > 2:
            sc.pp.highly_variable_genes(adata, n_top_genes=self.hparams.hvgs)
            self.data = torch.tensor(adata.X[:, adata.var["highly_variable"]].toarray())
            self.velocity = torch.tensor(
                adata.layers["velocity"][:, adata.var["highly_variable"]]
            )

        else:
            self.data = torch.from_numpy(adata.obsm["X_umap"])
            v_x, v_y = get_cell_velocities(adata)
            self.velocity = torch.stack([v_x, v_y], dim=1)

        if "fast_dev_run" in kwargs and kwargs["fast_dev_run"]:
            train_val_test_split = (1, 1, 1)

        if not isinstance(train_val_test_split, int) and isinstance(
            train_val_test_split[0], float
        ):
            # Split according to proportions
            assert np.isclose(sum(train_val_test_split), 1)
            train_frac, val_frac, test_frac = train_val_test_split
            train_num = round(train_frac * adata.shape[0])
            val_num = round(val_frac * adata.shape[0])
            test_num = adata.shape[0] - train_num - val_num
            assert train_num >= 0 and val_num >= 0 and test_num >= 0
            train_val_test_split = (train_num, val_num, test_num)

        # Create and split datasets
        self.split_dataset(train_val_test_split, [self.data, self.velocity])

    def split_dataset(self, train_val_test_split, tensors):
        dataset = TensorDataset(*tensors)

        if isinstance(train_val_test_split, int):
            self.data_train, self.data_val, self.data_test = dataset, dataset, dataset
        else:
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=train_val_test_split,
                generator=torch.Generator().manual_seed(self.seed),
            )
        
        train_x, train_vel = zip(*[self.data_train[i] for i in range(len(self.data_train))])
        self.train_x = torch.stack(train_x)
        self.train_vel = torch.stack(train_vel)
        
        # Wrap the datasets with PairedSamplingDataset
        self.paired_train = PairedSamplingDataset(self.data_train)
        self.paired_val = PairedSamplingDataset(self.data_val)
        self.paired_test = PairedSamplingDataset(self.data_test)

    @property
    def folder(self) -> str:
        return os.path.join(self.hparams.data_dir, self.__class__.__name__)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.paired_train,  # Use paired dataset
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.paired_val,  # Use paired dataset
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.paired_test,  # Use paired dataset
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def push_reference_to(self, device: torch.device) -> None:
        self.train_x = self.train_x.to(device, non_blocking=True)
        self.train_vel = self.train_vel.to(device, non_blocking=True)



if __name__ == "__main__":
    _ = DeepCycleDataModule()
