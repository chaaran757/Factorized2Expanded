from tqdm import tqdm
import numpy as np
import torch
from model import *

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

def train_epoch(model, train_iter, optimizer, loss_fn):
    model.train()
    losses = 0
    
    progress_bar = enumerate(tqdm(train_iter))
    
    for idx, (src, tgt) in progress_bar:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                                src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:,:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
        
    return losses / len(train_iter)
