import os
import torch
import numpy as np

def train_epoch(model, optimizer, train_loader, log, epoch, device, eval_batches=300):
    total_loss = 0
    total_kl = 0
    total_neg_logpx_z = 0
    num_batches = 0

    model.train()
    for x in train_loader:
        optimizer.zero_grad()
        x = x[0].float().to(device)

        x_batched = x.view(x.shape[0], -1) 
        z, probs_x, kl_z, neg_logpx_z = model(x_batched)

        avg_KLD = kl_z.sum() / x_batched.shape[0]
        avg_neg_logpx_z = neg_logpx_z.sum() / x_batched.shape[0]
        loss = avg_neg_logpx_z + avg_KLD

        loss.backward()
        optimizer.step()    

        total_loss += loss
        total_neg_logpx_z += avg_neg_logpx_z
        total_kl += avg_KLD
        num_batches += 1
        b_idx = epoch * len(train_loader) + num_batches

        if b_idx % eval_batches == 0:
            log('Train Total Loss', loss)
            log('Train -LogP(x|z)', avg_neg_logpx_z)
            log('Train KLD', avg_KLD)

    return total_loss, total_neg_logpx_z, total_kl, num_batches


def eval_epoch(model, val_loader, device, n_is_samples=100):
    total_loss = 0
    total_kl = 0
    total_neg_logpx_z = 0
    total_is_estimate = 0.0
    num_batches = 0

    model.eval()
    with torch.no_grad():
        for x in val_loader:
            x = x[0].float().to(device)

            x_batched = x.view(x.shape[0], -1) 
            z, probs_x, kl_z, neg_logpx_z = model(x_batched)
            avg_KLD = kl_z.sum() / x_batched.shape[0]
            avg_neg_logpx_z = neg_logpx_z.sum() / x_batched.shape[0]

            loss = avg_neg_logpx_z + avg_KLD
            total_loss += loss
            total_neg_logpx_z += avg_neg_logpx_z
            total_kl += avg_KLD

            is_estimate = model.get_IS_estimate(x_batched, n_samples=n_is_samples)
            total_is_estimate += is_estimate.sum() / x_batched.shape[0]

            num_batches += 1

    return total_loss, total_neg_logpx_z, total_kl, total_is_estimate, num_batches

