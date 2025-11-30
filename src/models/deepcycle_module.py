from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.components.single_marginal_utils import *
import torchcfm
from torchcfm.utils import plot_trajectories
import ot as pot
import torchdyn
from torchdyn.core import NeuralODE
from torchcfm.utils import *
from src.models.components.mmd_metrics import *


class DeepCycleModule(LightningModule):
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        net: torch.nn.Module,
        geodesic_net: torch.nn.Module,
        datamodule: LightningModule,
        compile: bool,
        sigma: float,
        alpha: float,
    ) -> None:
    
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.geodesic_net = geodesic_net
        self.datamodule = datamodule
        self.sigma = sigma
        self.alpha = alpha

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.x0 = list()
        self.x1 = list()
        self.v0 = list()
        # self.alpha = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.geodesic_net(x)

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
        if self.current_epoch < self.trainer.max_epochs//2:
            x0, x1, v0, v1 = batch
            self.datamodule.push_reference_to(x0.device)
            train_x = self.datamodule.train_x
            train_vel = self.datamodule.train_vel
            t = torch.rand(x0.shape[0]).type_as(x0)
            #Get xt_dot
            xt, mu_t_dot, eps = get_xt_xt_dot(t, x0, x1, self.geodesic_net, sigma = self.sigma)
            # Estimate u_xt
            ut = get_u_xt(xt, train_x, train_vel)
            # Compute loss
            cosine_loss = 1 - torch.nn.functional.cosine_similarity(ut, mu_t_dot).mean()
            l2_loss = torch.mean((ut - self.alpha*mu_t_dot) ** 2)
            loss = cosine_loss + l2_loss
        else:
            x0, x1, v0, v1 = batch
            self.datamodule.push_reference_to(x0.device)
            train_x = self.datamodule.train_x
            train_vel = self.datamodule.train_vel
            x0, x1 = coupling(x0, x1, x0.shape[0], train_x, train_vel, self.geodesic_net, sigma=self.sigma)
            t = torch.rand(x0.shape[0]).type_as(x0)
            xt, mu_t_dot, eps = get_xt_xt_dot(t, x0, x1, self.geodesic_net, sigma=self.sigma)
            vt = self.net(torch.cat([xt.detach(), t[:, None]], dim=-1))
            loss = torch.mean((vt - mu_t_dot.detach()) ** 2)

        return loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        
        loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        if self.current_epoch < self.trainer.max_epochs//2:
            self.log("train_cond/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        else:
            self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

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
        #self.val_loss(loss)
        
        x0, x1, v0, v1 = batch

        self.x0.append(x0)
        self.x1.append(x1)
        self.v0.append(v0)

        if self.current_epoch<self.trainer.max_epochs//2:
            loss = loss + 10**6

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        x0 = torch.cat(self.x0, dim=0)
        x1 = torch.cat(self.x1, dim=0)
        v0 = torch.cat(self.v0, dim=0)
        
        self.x0 = list()
        self.x1 = list()    
        self.v0 = list()

        self.datamodule.push_reference_to(x0.device)
        train_x = self.datamodule.train_x
        train_vel = self.datamodule.train_vel

        wrapped_model = TorchWrapperWithMetrics(self.net, train_x, train_vel) # removed alpha
        node = NeuralODE(
            wrapped_model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        with torch.no_grad():
            z0 = torch.cat([x0, torch.zeros_like(x0[:, :2])], dim=1)
            traj_augmented = node.trajectory(z0, t_span=torch.linspace(0, 1, 2),)

        traj = traj_augmented[..., :-2]
        cossin_traj = traj_augmented[..., -2]
        L2_traj = traj_augmented[..., -1]
        self.val_cos_integral = cossin_traj[-1].mean()       
        self.val_L2_integral = L2_traj[-1].mean()
        
        vt_0 = self.net(torch.cat([x0, torch.zeros_like(x0[:, :1])], dim=-1))

        self.val_w2_x = torchcfm.optimal_transport.wasserstein(traj[-1], x1)
        self.val_cos_v0 = (1 - torch.nn.functional.cosine_similarity(vt_0, v0)).mean()
        metrics_dict = compute_distribution_distances_with_prefix(traj[-1], x1, "val")
        self.val_mmd = metrics_dict['val/RBF_MMD'] 

        self.log("val/w2_x", self.val_w2_x, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/cos_v0", self.val_cos_v0, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/L2_integral", self.val_L2_integral, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mmd", self.val_mmd, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/cosdist_integral", self.val_cos_integral, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch)
        x0, x1, v0, v1 = batch
        
        self.x0.append(x0)
        self.x1.append(x1)
        self.v0.append(v0)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        x0 = torch.cat(self.x0, dim=0)
        x1 = torch.cat(self.x1, dim=0)
        v0 = torch.cat(self.v0, dim=0)
        
        self.x0 = list()
        self.x1 = list()    
        self.v0 = list()

        self.datamodule.push_reference_to(x0.device)
        train_x = self.datamodule.train_x
        train_vel = self.datamodule.train_vel

        wrapped_model = TorchWrapperWithMetrics(self.net, train_x, train_vel) # removed alpha
        node = NeuralODE(wrapped_model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        with torch.no_grad():
            z0 = torch.cat([x0, torch.zeros_like(x0[:, :2])], dim=1)
            traj_augmented = node.trajectory(z0, t_span=torch.linspace(0, 1, 2),)

        traj = traj_augmented[..., :-2]
        cossin_traj = traj_augmented[..., -2]
        L2_traj = traj_augmented[..., -1]
        self.test_cos_integral = cossin_traj[-1].mean()       
        self.test_L2_integral = L2_traj[-1].mean()
        
        vt_0 = self.net(torch.cat([x0, torch.zeros_like(x0[:, :1])], dim=-1))

        self.test_w2_x = torchcfm.optimal_transport.wasserstein(traj[-1], x1)
        self.test_cos_v0 = (1 - torch.nn.functional.cosine_similarity(vt_0, v0)).mean()
        metrics_dict = compute_distribution_distances_with_prefix(traj[-1], x1, "test")
        self.test_mmd = metrics_dict['test/RBF_MMD'] 

        self.log("test/w2_x", self.test_w2_x, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/cos_v0", self.test_cos_v0, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/L2_integral", self.test_L2_integral, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mmd", self.test_mmd, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/cosdist_integral", self.test_cos_integral, on_step=False, on_epoch=True, prog_bar=True)
        

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
        # params = self.parameters()
        optimizer = self.hparams.optimizer(params=params)
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = DeepCycleModule(None, None, None, None)
