# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 19:02:07 2018

@author: jeane
"""

import torch
from torch.autograd import Function
import torch.nn as nn

class M_Step(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, a_i, R, V, _lambda, beta_a, beta_u):
        """
            \mu^h_j = \dfrac{\sum_i r_{ij} V^h_{ij}}{\sum_i r_{ij}}
            (\sigma^h_j)^2 = \dfrac{\sum_i r_{ij} (V^h_{ij} - mu^h_j)^2}{\sum_i r_{ij}}
            cost_h = (\beta_u + log \sigma^h_j) * \sum_i r_{ij}
            a_j = logistic(\lambda * (\beta_a - \sum_h cost_h))

            Input:
                a_i:      (b, B, 1)
                R:         (b, B, C)
                v:         (b, B, C, P*P)
            Local:
                cost_h:    (b, C, P*P)
                R_sum:     (b, C, 1)
            Output:
                a_j:     (b, C, 1)
                mu:        (b, 1, C, P*P)
                sigma_sq:  (b, 1, C, P*P)
        """
        b, B, C, psize = V.size()
        eps = 1e-8
        R_a = R * a_i
        #R = R / (R.sum(dim=2, keepdim=True) + eps)
        R_sum = R_a.sum(dim=1, keepdim=True)
        coeff = R_a / (R_sum + eps)
        coeff = coeff.view(b, B, C, 1)
        
        mu = torch.sum(coeff * V, dim=1, keepdim=True)
        sigma_sq = torch.sum(coeff * (V - mu)**2, dim=1, keepdim=True) + eps
        
        R_sum = R_sum.view(b, C, 1)
        sigma_sq = sigma_sq.view(b, C, psize)
        sigma = sigma_sq.sqrt()
        cost_h = (beta_u.view(C, 1) + torch.log(sigma)) * R_sum
        
        a_j = nn.Sigmoid()(_lambda*(beta_a - cost_h.sum(dim=2)))
        sigma_sq = sigma_sq.view(b, 1, C, psize)
        
        nu = (R_a[:,:,:,None]*V).sum(1)
        delta = V-nu[:,None,:,:]/R_sum[:,None,:,:]
        ctx.save_for_backward(cost_h, R_sum, R, R_a, _lambda, a_i, a_j, sigma, sigma_sq, V, mu, nu, delta)

        return a_j, mu, sigma_sq

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_out_a_j, grad_out_mu, grad_out_sigma_sq):
        grad_out_a_j_idx = torch.nonzero(grad_out_a_j)
        grad_out_mu_idx = torch.nonzero(grad_out_mu)
        grad_out_sigma_sq_idx = torch.nonzero(grad_out_sigma_sq)
        cost_h, R_sum, R, R_a, _lambda, a_i, a_j, sigma, sigma_sq, V, mu, nu, delta = ctx.saved_tensors
        
        if grad_out_a_j_idx.size(0) != 0:
            b, C = grad_out_a_j_idx[0]
            
            grad_cost_h_wrt_R_a = (cost_h/R_sum)[:,None,:,:]+1/(2*sigma_sq)*(V**2-((R_a[:,:,:,None]*(V**2)).sum(1)/R_sum)[:,None,:,:]-(2*nu/R_sum)[:,None,:,:]*delta)
            grad_cost_h_wrt_R_a = (grad_cost_h_wrt_R_a).sum(3)
            grad_R_a_wrt_a_i = R
            grad_a_j_wrt_cost_h = (a_j*_lambda*(a_j-1))
            
            grad_cost_h_wrt_sigma = (1/sigma*R_sum)
            grad_sigma_wrt_V = 1/sigma[:,None,:,:]*R_a[:,:,:,None]*delta/R_sum[:,None,:,:]
            
            grad_cost_h_wrt_beta_u = V.size(3)*R_sum.squeeze()
            grad_R_a_wrt_R = a_i
            
            grad_a_j_wrt_a_i = grad_out_a_j[:,None,:]*grad_a_j_wrt_cost_h[:,None,:]*grad_cost_h_wrt_R_a*grad_R_a_wrt_a_i
            grad_a_j_wrt_a_i = grad_a_j_wrt_a_i[:,:,C].unsqueeze(2)
            grad_a_j_wrt_V = grad_out_a_j[:,None,:,None]*grad_a_j_wrt_cost_h[:,None,:,None]*grad_cost_h_wrt_sigma[:,None,:,:]*grad_sigma_wrt_V
            grad_a_j_wrt_beta_a = (-grad_out_a_j*grad_a_j_wrt_cost_h)[b,:]
            grad_a_j_wrt_beta_u = (grad_out_a_j*grad_a_j_wrt_cost_h*grad_cost_h_wrt_beta_u)[b,:]
            grad_a_j_wrt_R = (grad_out_a_j*grad_a_j_wrt_cost_h)[:,None,:]*grad_cost_h_wrt_R_a*grad_R_a_wrt_R

            return grad_a_j_wrt_a_i, grad_a_j_wrt_R, grad_a_j_wrt_V, None, grad_a_j_wrt_beta_a, grad_a_j_wrt_beta_u
        elif grad_out_mu_idx.size(0) != 0:
            b, _, C,P = grad_out_mu_idx[0]
            
            grad_mu_wrt_R_a = (((V[:,:,:,P]-mu[b,:,C,P])[:,:,:,None]*grad_out_mu)[:,:,:,P]/R_sum.squeeze()[:,None,:])
            
            grad_mu_wrt_a_i = grad_mu_wrt_R_a[:,:,C]*R[:,:,C]
            grad_mu_wrt_R = grad_mu_wrt_R_a*a_i
            grad_V = (R_a[:,:,:,None]*grad_out_mu/R_sum[:,None,:,:]).expand(V.size())
            
            return grad_mu_wrt_a_i.unsqueeze(2), grad_mu_wrt_R, grad_V, None, None, None
        elif grad_out_sigma_sq_idx.size(0) != 0:
            b, B, C, P = grad_out_sigma_sq_idx[0]
            
            grad_sigma_sq_wrt_R_a = V**2/R_sum[:,None,:,:]-((R_a[:,:,:,None]*(V**2)).sum(1)/(R_sum**2))[:,None,:,:]-(2*nu/(R_sum**2))[:,None,:,:]*delta
            grad_sigma_sq_wrt_a_i = (grad_sigma_sq_wrt_R_a*grad_out_sigma_sq)[:,:,C,P]*R[:,:,C]
            grad_sigma_sq_wrt_R = (grad_out_sigma_sq*grad_sigma_sq_wrt_R_a)[:,:,:,P]*a_i
            
            grad_V = 2*grad_out_sigma_sq/R_sum[:,None,C,:][:,None,:,:]*delta[:,:,C,P][:,:,None,None]*R_a[:,:,C,None][:,:,:,None]
            
            return grad_sigma_sq_wrt_a_i.unsqueeze(2), grad_sigma_sq_wrt_R, grad_V, None, None, None
        else:
            return None, None, None, None, None, None

mstep = M_Step.apply

torch.manual_seed(1)

b, B, C, P = 8, 8, 16, 4

a_in = torch.randn(b, B, 1, requires_grad=True).abs().cuda().double()/B
R = torch.ones(b, B, C, requires_grad=True).abs().cuda().double()/C
V = torch.randn(b, B, C, P*P, requires_grad=True).cuda().double()/(P*P)

beta_u = (torch.ones(C, requires_grad=True)/C).cuda().double()
beta_a = (torch.ones(C, requires_grad=True)/C).cuda().double()

from torch.autograd import gradcheck

input = (a_in, R, V, torch.tensor(1e-4).cuda().double(), beta_a, beta_u)

test = gradcheck(M_Step.apply, input, eps=1e-6, atol=1e-6, rtol=1e-2, raise_exception=True)
print(test)
