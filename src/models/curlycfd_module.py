from typing import Any, Dict, Tuple, Optional

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

import torchcfm
from torchcfm.utils import plot_trajectories
import ot as pot
import torchdyn
from torchdyn.core import NeuralODE
from torchcfm.utils import *
from src.models.components.mmd_distr import *
from torchcfm.optimal_transport import OTPlanSampler
from models.components.cfd_utils import *
from torch.utils.data import DataLoader


class CurlyCFDModule(LightningModule):
    
    def __init__(
        self,
        geo_model: torch.nn.Module,
        vel_model: torch.nn.Module,
        score_model: torch.nn.Module,
        sigma: float,
        k: int, 
        n_times: float, 
        alpha: float,
        num_times: int,
        datamodule: LightningModule,
        compile: bool,
    ) -> None:
    
        super().__init__()

        self.automatic_optimization = False

        self.save_hyperparameters(logger=False)

        self.geo_model = geo_model
        
        self.vel_model = vel_model

        self.score_model = score_model

        self.sigma = sigma

        self.k = k

        self.n_times = n_times

        self.alpha = alpha

        self.num_times = num_times

        self.datamodule = datamodule

        self._bs_switched = False

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.geo_model(x)

    def on_train_start(self) -> None:
        self.val_loss.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Unpack the batch
        batch = batch[0]
        timesteps = torch.linspace(0.0, 1, len(batch)).tolist()

        batch_x0, batch_x1 = batch[:-1], batch[1:] 
        
        x0s = [x[0] for x in batch_x0]
        x1s = [x[0] for x in batch_x1]
        v0s = [x[1] for x in batch_x0]
        v1s = [x[1] for x in batch_x1]

        if self.current_epoch < self.trainer.max_epochs // 2: 
            self.geo_optimizer.zero_grad()
            t_orig, t, xt, ut, mu_t_dot, eps = get_batch_geo(x0s, x1s, v0s, v1s, timesteps)
            cosine_loss = 1 - torch.nn.functional.cosine_similarity(ut, mu_t_dot).mean()
            loss = torch.mean((self.alpha*ut - mu_t_dot) ** 2) + cosine_loss
            
            loss.backward()
            self.geo_optimizer.step()
        else:
            self.sf2m_optimizer.zero_grad()
            t_orig, t, xt, mu_t_dot, _, _, eps = get_batch_vel(x0s, x1s, v0s, v1s, timesteps)
            vt = self.vel_model(torch.cat([xt.detach(), t[:, None]], dim=-1))
            st = self.score_model(torch.cat([xt, t[:, None]], dim=-1))
            
            lambda_t = (2* torch.sqrt(t_orig * (1-t_orig))) / self.sigma
            
            flow_loss = torch.mean((vt - mu_t_dot.detach()) ** 2)
            
            score_loss = torch.mean((lambda_t[:, None] * st + eps) ** 2)
            loss = flow_loss + score_loss

            loss.backward()
            self.sf2m_optimizer.step()

        return loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        if self.current_epoch < self.trainer.max_epochs // 2:  
            self.log("train_geonet/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        else:
            self.log("train_vel/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        pass

    def on_validation_epoch_end(self) -> None:
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        batch = batch[0]
        timesteps = torch.linspace(0, 1, len(batch)).tolist()

        cos_dists = []
        w2s = []
        cos_integrals = []
        cos_integral_points = []
        l2s = []
        mse_list = []
        presicion_at_k = []
        
        for i in range(len(batch)-1):
            t_start = timesteps[i]
            t_end= timesteps[i+1]
            batch_0 = batch[i]
            batch_1 = batch[i+1]
            x0 = batch_0[0]
            x1 = batch_1[0]
            v0 = batch_0[1]
            v1 = batch_1[1]

            wrapped_model = CurlyWrapperWithMetricsCFD(self.vel_model, x0, x1, v0, v1, self.k)
            node = NeuralODE(wrapped_model, solver="dopri5", sensitivity="adjoint")
            z0 = torch.cat([x0, torch.zeros(x0.shape[0], 3, device=x0.device)], dim=1)

            traj_augmented = node.trajectory(
                z0,
                t_span=torch.linspace(
                    t_start, t_end, 2
                ),
            )
            traj = traj_augmented[..., :-3]
            cossin_traj_all = traj_augmented[..., -3]
            cossin_traj = traj_augmented[..., -2]
            L2_traj = traj_augmented[..., -1]
            
            X_mid_pred = traj[-1]

            cos_integral = cossin_traj_all[-1].mean()
            cos_integral_point = cossin_traj[-1].mean()       
            L2_integral = L2_traj[-1].mean()

            #w2 metric computation
            w2_metric = torchcfm.optimal_transport.wasserstein(X_mid_pred, x1)

            #marginal velocity comparison
            t_dot = torch.tensor(t_end).to(self.device)
            x1_dot = self.vel_model((torch.cat([x1, t_dot.repeat(x1.shape[0])[:, None]], dim=1)))
            cos_dist = 1 - torch.nn.functional.cosine_similarity(x1_dot, v1).mean()

            # MSE between predicted and true x1
            mse = torch.nn.functional.mse_loss(X_mid_pred, x1)

            # Hit@k for how often the predicted x1 is within k nearest neighbors of the true x1
            # compute pairwise distances between predicted and ground truth
            dists = torch.cdist(X_mid_pred, x1)  # shape: (N, N) where N is the number of particles
            targets = torch.arange(x1.size(0), device=x1.device).unsqueeze(1)  # shape: (N, 1)

            # compute hit@k for multiple k
            k_list = [1, 5, 10, 25, 50, 100]  
            test_hit_at_k = {}
            for k in k_list:
                topk = torch.topk(dists, k, largest=False, dim=1).indices  # (N, k)
                hits = (topk == targets).any(dim=1).float()  # (N,)
                test_hit_at_k[k] = hits.mean()  # average over batch

            cos_dists.append(cos_dist)
            w2s.append(w2_metric)
            cos_integrals.append(cos_integral)
            cos_integral_points.append(cos_integral_point)
            l2s.append(L2_integral)

            mse_list.append(mse)
            presicion_at_k.append(test_hit_at_k)
        
        self.test_cos_v = torch.stack(cos_dists).mean()
        self.test_w2_x = sum(w2s) / len(w2s)
        self.test_cos_integral = torch.stack(cos_integrals).mean()
        self.test_cos_integral_point = torch.stack(cos_integral_points).mean()
        self.test_L2_integral = torch.stack(l2s).mean()

        self.test_mse = sum(mse_list) / len(mse_list)

        self.test_hit_at_k = {}
        for k in k_list:
            self.test_hit_at_k[k] = sum([x[k] for x in presicion_at_k]) / len(presicion_at_k)
        
        self.log("test/w2_x", self.test_w2_x, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/L2", self.test_L2_integral, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/cos_dist_int", self.test_cos_integral, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/cos_dist_int_point", self.test_cos_integral_point, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/cos_dist_v", self.test_cos_v, on_step=False, on_epoch=True, prog_bar=True)

        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True, prog_bar=True)
        for k in k_list:
            self.log(f"test/hit_at_{k}", self.test_hit_at_k[k], on_step=False, on_epoch=True, prog_bar=True)
         
    def on_test_epoch_end(self) -> None:
        pass
        

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.vel_model = torch.compile(self.vel_model)

    def configure_optimizers(self) -> Dict[str, Any]:
        self.geo_optimizer = torch.optim.AdamW(self.geo_model.parameters(), 1e-4)
        self.vel_optimizer = torch.optim.Adam(self.vel_model.parameters(), 1e-4)
        self.sf2m_optimizer = torch.optim.AdamW(
            list(self.vel_model.parameters()) + list(self.score_model.parameters()), 5e-4
        )
        return self.geo_optimizer, self.vel_optimizer, self.sf2m_optimizer


if __name__ == "__main__":
    _ = CurlyCFDModule(None, None, None, None)
