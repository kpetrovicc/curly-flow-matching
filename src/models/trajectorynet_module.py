from typing import Any, Dict, Tuple

import numpy as np
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.components.utils import *
import torchcfm
from torchcfm.utils import plot_trajectories
from src.models.components.augmentation import (
    AugmentationModule,
    AugmentedVectorField,
    Sequential,
    SquaredL2Reg,
)
import ot as pot
import torchdyn
from torchdyn.core import NeuralODE
from torchcfm.utils import *
from torch.distributions import MultivariateNormal

class TrajectoryNetModule(LightningModule):
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        net: torch.nn.Module,
        geodesic_net: torch.nn.Module,
        augmentations: AugmentationModule,
        datamodule: LightningModule,
        compile: bool,
        leaveout_timepoint: int = -1,
    ) -> None:
    
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.geodesic_net = geodesic_net
        
        self.datamodule = datamodule
        
        self.dim = self.datamodule.dim
       
        self.augmentations = augmentations
        self.augmentations.regs[1].ut = lambda x: get_u_xt(x, self.datamodule.train_x, self.datamodule.train_vel) # hard coded currently to select the velocity reg
        self.aug_net = AugmentedVectorField(
            torch_wrapper(self.net), self.augmentations.regs, self.dim
        )
        self.val_augmentations = AugmentationModule(
            # cnf_estimator=None,
            l1_reg=1,
            l2_reg=1,
            squared_l2_reg=1,
            squared_l2_vector_reg=1,
        )
        self.val_aug_net = AugmentedVectorField(
            self.net, self.val_augmentations.regs, self.dim
        )
        self.aug_node = Sequential(
            self.augmentations.augmenter,
            NeuralODE(
                self.aug_net,
                solver="euler",
                #sensitivity="adjoint",
                atol=1e-4,
                rtol=1e-4,
            ),
        )

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.geodesic_net(x)
    
    def forward_integrate(self, batch: Any, t_span: torch.Tensor):
        """Forward pass with integration over t_span intervals.

        (t, x, t_span) -> [x_t_span].
        """
        X = self.unpack_batch(batch)
        X_start = X[:, t_span[0], :]
        traj = self.node.trajectory(X_start, t_span=t_span)
        return traj

    def step(self, batch: Any, training: bool = False):
        pass

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
    
        x0, x1, v0, v1 = batch
        
        even_ts = torch.arange(2).to(x0) + 1 # hard coded for t0, t1 curently
        
        self.prior = MultivariateNormal(
            torch.zeros(x0.shape[-1]).type_as(x0), torch.eye(x0.shape[-1]).type_as(x0)
        )
        # Minimize the log likelihood by integrating all back to the initial timepoint
        reversed_ts = torch.cat(
            [torch.flip(even_ts, [0]), torch.tensor([0]).type_as(even_ts)]
        )
    
        losses = []
        regs = []
        for t in range(len(reversed_ts) - 1):
            ts = reversed_ts[t:]
            xs = [x1, x0]
            ts = torch.linspace(ts[0], ts[-1], 20) # ts[0] = t1, ts[-1] = gaussian
            _, x = self.aug_node(xs[t], ts)
            x = x[-1]
            # Assume log prob is in zero spot
            delta_logprob, reg, x = self.augmentations(x)
            logprob = self.prior.log_prob(x).to(x) - delta_logprob
            losses.append(-torch.mean(logprob))

            regs.append(-reg)
            
        reg = torch.mean(torch.stack(regs))
        loss = torch.mean(torch.stack(losses))
        return reg, loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        
        reg, mse = self.model_step(batch)
        loss = mse + reg

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        x0, x1, v0, v1 = batch

        if (self.current_epoch+1) % 100 == 0:
            node = NeuralODE(
            torch_wrapper(self.net), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
            )
            with torch.no_grad():
                traj = node.trajectory(
                    x0,
                    t_span=torch.linspace(0, 1, 100),
                )
                # torch.save(traj, os.path.join('traj', f"traj_{self.current_epoch}.pt"))
                #plot_trajectories_f(traj, x1,'traj', self.current_epoch)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch)
        x0, x1, v0, v1 = batch
        # update and log metrics
        self.val_loss(loss)

        vel = x1 - x0

        node = NeuralODE(
            torch_wrapper(self.net), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        with torch.no_grad():
            traj = node.trajectory(
                x0,
                t_span=torch.linspace(1, 2, 2),
            )
        
        self.val_w2_x = torchcfm.optimal_transport.wasserstein(traj[-1], x1)
        self.val_cos_v = 1 - torch.nn.functional.cosine_similarity(vel, v1).mean()

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/w2_x", self.val_w2_x, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/cos_v", self.val_cos_v, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch)
        self.test_loss(loss)
        x0, x1, v0, v1 = batch

        vel = x1 - x0

        node = NeuralODE(
            torch_wrapper(self.net), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        with torch.no_grad():
            traj = node.trajectory(
                x0,
                t_span=torch.linspace(1, 2, 2),
            )
        
        self.test_w2_x = torchcfm.optimal_transport.wasserstein(traj[-1], x1)
        self.test_cos_v = 1 - torch.nn.functional.cosine_similarity(vel, v1).mean()

        # update and log metrics
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/w2_x", self.test_w2_x, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/cos_v", self.test_cos_v, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        params = list(self.net.parameters()) + list(self.geodesic_net.parameters())
        optimizer = self.hparams.optimizer(params=params)
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = TrajectoryNetModule(None, None, None, None)
