import torch
import matplotlib.pyplot as plt
import scipy 
import os
import torch.nn.functional as F
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde
import ot as pot
import anndata as ad
import matplotlib.colors as mcolors
import pandas as pd
import scanpy as sc
import scvelo as scv
from models.components.curlyfm.kernel_factory import *
from functools import partial
from torch.func import vmap

def compute_gamma(t, t_min, t_max):
    return (
            1.0
            - ((t - t_min) / (t_max - t_min)) ** 2
            - ((t_max - t) / (t_max - t_min)) ** 2
        )

def get_xt(t, t_min, t_max, x0, x1, geodesic_model, sigma=0.0):
    gamma = compute_gamma(t, t_min, t_max)
    mu_t = ((t_max - t) / (t_max - t_min)) * x0 + ((t - t_min) / (t_max - t_min)) * x1 + gamma * (geodesic_model(torch.cat([x0, x1, t], dim=-1))) 
    epsilon = torch.randn_like(x0)
    x_t = mu_t + torch.sqrt(t*(1-t))*sigma * epsilon
    return mu_t, x_t, epsilon

def get_xt_xt_dot(t, t_start, t_end, x0, x1, geodesic_model, sigma=0.0):
    # get xt and xt dot from the geodesic model
    with torch.enable_grad():
        t = t[..., None]
        t.requires_grad_(True)
        mu_t, xt, eps = get_xt(t, t_start, t_end, x0, x1, geodesic_model, sigma=sigma)
        mu_t_dot_list = []
        for i in range(xt.shape[-1]):
            mu_t_dot_list.append(
                torch.autograd.grad(torch.sum(mu_t[..., i]), t, create_graph=True)[0]
            )
        # this is velocity (euclidean metric)
        mu_t_dot = torch.cat(mu_t_dot_list, -1)
    return xt, mu_t_dot, eps

def get_ut_knn_gaussian(
    xt: torch.Tensor,      
    x0: torch.Tensor,      
    x1: torch.Tensor,      
    v0: torch.Tensor,      
    v1: torch.Tensor,      
    k:  int = 100,
    eps: float = 1e-12,
):
    x = torch.cat([x0, x1], dim=0)
    v = torch.cat([v0, v1], dim=0)
    dists = torch.cdist(xt, x)
    knn_dists, knn_idx = torch.topk(dists, k=k, dim=1, largest=False)
    h = knn_dists[:, -1:].clamp_min(eps)       

    # Gaussian kernel weights
    w = torch.exp(- (knn_dists**2) / (2 * h**2))      
    w = w / (w.sum(dim=1, keepdim=True) + eps) 

    # gather neighbour velocities and form weighted sum
    v_knn = v[knn_idx]                         
    v_xt  = (w.unsqueeze(-1) * v_knn).sum(dim=1)

    return v_xt

def coupling_cfd(t_start, t_end, x0, x1, x0s, x1s, v0s, v1s, geodesic_model, k, sigma, num_times=1):

    L2_cost_tot = 0
    batch_size, d = x0.shape

    if num_times == 1:
        t = torch.rand(1).type_as(x0)
        t = t * (t_end - t_start) + t_start
        t = t * torch.ones(batch_size, batch_size, device=x0.device)
        t_tensor = t.unsqueeze(0)
    else:
        times = torch.linspace(t_start, t_end, num_times).type_as(x0)
        t_tensor = times[:, None, None].repeat(1, batch_size, batch_size)
    
    x0_r = x0.repeat(batch_size, 1, 1)
    x1_r = x1.repeat(batch_size, 1, 1).transpose(0, 1)

    for i in range(num_times):
        t = t_tensor[i]

        xt, mu_t_dot, eps = get_xt_xt_dot(t, t_start, t_end, x0_r, x1_r, geodesic_model, sigma)  

        ut = vmap(lambda x: get_ut_knn_gaussian(x, x0s, x1s, v0s, v1s, k=k))(xt)

        L2_cost = 0.5 * ((mu_t_dot.detach() - ut) ** 2).sum(-1)   
        L2_cost_tot += L2_cost

    _, j = scipy.optimize.linear_sum_assignment(L2_cost_tot.detach().cpu().numpy())

    pi_x0 = x0[j]          
    pi_x1 = x1             

    return pi_x0, pi_x1, eps

class CurlyWrapperWithMetricsCFD(torch.nn.Module):
    def __init__(self, model, x0, x1, v0, v1, k):
        super().__init__()
        self.model = model
        self.x0 = x0
        self.x1 = x1
        self.v0 = v0
        self.v1 = v1
        self.k = k
        self.cos_dist_all = 0
    
    def forward(self, t, z, *args, **kwargs):
        x = z[:, :-3]
        x_dot = self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], dim=1))
        
        u_t = get_ut_knn_gaussian(
                x,
                self.x0,
                self.x1,
                self.v0,
                self.v1,
                k=self.k,
            )
        
        cos_dist = 1 - F.cosine_similarity(u_t, x_dot, dim=1)
        self.cos_dist_all+= cos_dist
        L2_squared = torch.sum((u_t - x_dot) ** 2, dim=1)

        return torch.cat([x_dot, self.cos_dist_all.unsqueeze(1), cos_dist.unsqueeze(1), L2_squared.unsqueeze(1)], dim=1)

def get_batch_geo(self, x0s, x1s, v0s, v1s, timesteps):
        t_orig_list = []
        ts = []
        xts = []
        uts = []
        mu_t_dots = []
        eps_list = []
        t_start = timesteps[0]
        
        for i, (x0, x1, v0, v1) in enumerate(zip(x0s, x1s, v0s, v1s)):
            x0, x1= torch.squeeze(x0), torch.squeeze(x1)
            v0, v1 = torch.squeeze(v0), torch.squeeze(v1)
            t_start_next = timesteps[i + 1]

            t = torch.rand(x0.shape[0]).type_as(x0) 
            t_o = t
            t = t * (t_start_next - t_start) + t_start
            
            xt, mu_t_dot, eps = get_xt_xt_dot(t, t_start, t_start_next, x0, x1, self.geo_model, sigma=self.sigma)
            ut = get_ut_knn_gaussian(
                xt,
                x0,
                x1,
                v0,
                v1,
                k=self.k,
            )
            
            xt = xt + torch.sqrt(t* (1-t)).unsqueeze(1) * self.sigma * eps
            
            t_orig_list.append(t_o)
            ts.append(t)
            xts.append(xt)
            uts.append(ut)
            mu_t_dots.append(mu_t_dot)
            eps_list.append(eps)
            t_start = t_start_next
        
        t_orig = torch.cat(t_orig_list)
        t = torch.cat(ts)
        xt = torch.cat(xts)
        ut = torch.cat(uts)
        mu_t_dot = torch.cat(mu_t_dots)
        eps = torch.cat(eps_list)

        return t_orig, t, xt, ut, mu_t_dot, eps

def get_batch_vel(self, x0s, x1s, v0s, v1s, timesteps):
    t_orig = []
    ts = []
    xts = []
    mu_t_dots = []
    eps_list = []
    t_start = timesteps[0]

    for i, (x0, x1, v0, v1) in enumerate(zip(x0s, x1s, v0s, v1s)):
        x0, x1= torch.squeeze(x0), torch.squeeze(x1)
        v0, v1 = torch.squeeze(v0), torch.squeeze(v1)

        t_start_next = timesteps[i + 1]

        t = torch.rand(x0.shape[0]).type_as(x0)
        t_o = t
        t = t * (t_start_next - t_start) + t_start
        x0, x1, _ = coupling_cfd(t_start, t_start_next, x0, x1, x0, x1, v0, v1, self.geo_model, self.k, sigma = self.sigma, num_times=self.num_times)       
        xt, mu_t_dot, eps = get_xt_xt_dot(t, t_start, t_start_next, x0, x1, self.geo_model, sigma=self.sigma)
        t_orig.append(t_o)
        ts.append(t)
        xts.append(xt)
        mu_t_dots.append(mu_t_dot)
        eps_list.append(eps)
        t_start = t_start_next

    t = torch.cat(ts)
    t_orig = torch.cat(t_orig)
    xt = torch.cat(xts)
    mu_t_dot = torch.cat(mu_t_dots)
    eps = torch.cat(eps_list)

    return t_orig, t, xt, mu_t_dot, v0, v1, eps
