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

def greedy_decode(model, src, src_mask, max_len, start_symbol, mapping_dict):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    
    for i in range(max_len-1):
        memory = memory.to(device)
        memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == mapping_dict['end']:
            break
    return ys


def expand(model, src, mapping_dict, inv_mapping_dict):
    model.eval()
    num_tokens = len(src)
    
    src = (torch.LongTensor(src).reshape(num_tokens, 1) )
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(model, src, src_mask, 29, mapping_dict['start'], mapping_dict).flatten().cpu().numpy()
    
    return " ".join([inv_mapping_dict[tok] for tok in tgt_tokens]).replace("start", "").replace("end", "")
