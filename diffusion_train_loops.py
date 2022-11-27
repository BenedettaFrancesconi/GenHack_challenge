import os
import torch
import numpy as np

def train_epoch_diffusion(model, optimizer, train_loader, log, epoch, device, eval_batches=300):
    total_loss = 0
    num_batches = 0

    model.train()
    for x in train_loader:
        optimizer.zero_grad()
        x = x[0].float().to(device)

        x_batched = x.view(x.shape[0], -1, 1) # (batch_size, n_channels, seq_length)
        loss = model(x_batched)
        loss.backward()

        optimizer.step()    

        total_loss += loss

        num_batches += 1
        b_idx = epoch * len(train_loader) + num_batches

        if b_idx % eval_batches == 0:
            log('Train Total Loss', loss)

    return total_loss, num_batches


def eval_epoch_diffusion(model, val_loader, device):
    total_loss = 0
    num_batches = 0

    model.eval()
    with torch.no_grad():
        for x in val_loader:
            x = x[0].float().to(device)

            x_batched = x.view(x.shape[0], -1, 1) 
            loss = model(x_batched)
            total_loss += loss
            num_batches += 1

    return total_loss, num_batches



