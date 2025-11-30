import torch
import matplotlib.pyplot as plt
import scipy 
import os
import torch.nn.functional as F
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde
import ot as pot
from torch.func import vmap
import seaborn as sns
    
class TorchWrapperWithMetrics(torch.nn.Module):
    def __init__(self, model, data, velocity):
        super().__init__()
        self.model = model
        self.data = data
        self.velocity = velocity
    
    def forward(self, t, z, *args, **kwargs):
        x = z[:, :-2]
        metric = z[:, -2:]

        batch_size = x.shape[0]
        x_dot = self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], dim=1))
        
        u_t = get_u_xt(x, self.data, self.velocity)
        
        cos_sim = 1 - F.cosine_similarity(u_t, x_dot, dim=1)
        L2_squared = torch.sum((u_t - x_dot) ** 2, dim=1)
                
        return torch.cat([x_dot, cos_sim.unsqueeze(1), L2_squared.unsqueeze(1)], dim=1)

def get_cell_velocities(adata):
    velocities = adata.obsm['velocity_umap']
    V_x = torch.from_numpy(velocities[:, 0]).float()  # x velocities
    V_y = torch.from_numpy(velocities[:, 1]).float()  # y velocities
    return V_x, V_y

def get_xt(t, x0, x1, geodesic_model, sigma=0.0):
    mu_t = (1 - t) * x0 + t * x1 +  t * (1-t) * (geodesic_model(torch.cat([x0, x1, t], dim=-1)))
    epsilon = torch.randn_like(x0)
    x_t = mu_t + torch.sqrt(t*(1-t))*sigma * epsilon
    return mu_t, x_t, epsilon

def get_xt_xt_dot(t, x0, x1, geodesic_model, sigma=0.0):
    with torch.enable_grad():
        t = t[..., None]
        t.requires_grad_(True)
        mu_t, xt, eps = get_xt(t, x0, x1, geodesic_model, sigma=sigma)
        mu_t_dot_list = []
        for i in range(xt.shape[-1]):
            mu_t_dot_list.append(
                torch.autograd.grad(torch.sum(mu_t[..., i]), t, create_graph=True)[0]
            )
        mu_t_dot = torch.cat(mu_t_dot_list, -1)
    return xt, mu_t_dot, eps

def get_u_xt(xt, x, v, k=20):
    dists = torch.cdist(xt, x)
    knn_dists, knn_idx = torch.topk(dists, k=k, dim=1, largest=False)
    h = knn_dists[:, -1:].clamp_min(1e-12)  
    w = torch.exp(-(knn_dists**2) / (2 * h**2))  
    w = w / (w.sum(dim=1, keepdim=True) + 1e-12)  
    v_knn = v[knn_idx]  
    v_xt = (w.unsqueeze(-1) * v_knn).sum(dim=1)

    return v_xt

def coupling(x0, x1, batch_size, xs, vs, geodesic_model, sigma):
    t = torch.rand(1).type_as(x0) * torch.ones(batch_size, batch_size, device=x0.device)
    x0_r = x0.repeat(batch_size, 1, 1)
    x1_r = x1.repeat(batch_size, 1, 1).transpose(0, 1)
    xt, mu_t_dot, eps = get_xt_xt_dot(t, x0_r, x1_r, geodesic_model, sigma=sigma)
    ut = vmap(lambda x: get_u_xt(x, xs, vs))(xt)
    L2_cost = 0.5*((mu_t_dot.detach() - ut)**2).sum(-1)
    _, col_ind = scipy.optimize.linear_sum_assignment(L2_cost.detach().cpu().numpy())
    pi_x0 = x0[col_ind]
    pi_x1 = x1
    return pi_x0, pi_x1

def sample_ot(x0, x1): # taken from torchcfm repo 
        batch_size = x0.shape[0]
        a, b = pot.unif(x0.size()[0]), pot.unif(x1.size()[0])
        M = torch.cdist(x0, x1) ** 2
        pi = pot.emd(a, b, M.detach().cpu().numpy())
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=batch_size)
        i, j = np.divmod(choices, pi.shape[1])
        x0 = x0[i]
        x1 = x1[j]
        return x0, x1

def sample_conditional_pt(x0, x1, t, sigma):
    t = t.reshape(-1, *([1] * (x0.dim() - 1)))
    mu_t = t * x1 + (1 - t) * x0
    epsilon = torch.randn_like(x0)
    return mu_t + sigma * epsilon

def plot_trajectories(traj):
    """Plot trajectories of some selected samples."""
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")
    plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    plt.xticks([])
    plt.yticks([])
    plt.show()

def plot_x(x0, x1):
    sns.set_style("white")
    plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.set_frame_on(False)
    sns.scatterplot(
            x=x0[:, 0], 
            y=x0[:, 1],
            alpha=0.5,
            label="$q_{(0)}$",
            color='blue',
            s=30
        )
    sns.scatterplot(
            x=x1[:, 0], 
            y=x1[:, 1],
            alpha=0.5,
            label="$q_{(1)}$",
            color='#E8B088',
            s=30
        )
        
    # Remove axes and frame
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')    
    # Keep aspect ratio equal
    plt.axis('equal')
    plt.show()

def plot_trajectories_2(traj):
    n = 2000
    sns.set_theme(style="white", context="notebook")
    fig = plt.figure(figsize=(6, 6), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.set_frame_on(False)
    timesteps, n_samples, _ = traj.shape
    timesteps = traj.shape[0]
    colors = plt.cm.rainbow(np.linspace(0, 1, timesteps))
    sns.scatterplot(x=traj[:, :, 0].flatten(),
                   y=traj[:, :, 1].flatten(),
                   c='#B5B5E9',
                   s=8,
                   linewidth=0,
                   alpha=0.6)
    sns.scatterplot(x=traj[0, :, 0], y=traj[0, :, 1],
                   color='blue', s=70, alpha=0.8,
                   label="$x_{(0)}$", zorder=2)
    sns.scatterplot(x=traj[-1, :, 0], y=traj[-1, :, 1],
                   color='#E8B088', s=70, alpha=0.8,
                   label="$\hat{x}_{(1)}$", zorder=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.legend(fontsize=16, frameon=True)
    
    plt.show()