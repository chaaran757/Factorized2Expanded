import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

class EmbeddingDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embed, words, p):
        mask = torch.from_numpy(np.random.binomial(1,p,size=(embed.weight.data.shape[0]))/p).to(device)
        mask = Variable(mask, requires_grad=False)
        masked_embedding_weights = mask.unsqueeze(1) * embed.weight.data
        masked_embedding_weights = masked_embedding_weights.to(device)
        embedding = torch.nn.functional.embedding(words.long(), masked_embedding_weights, padding_idx=0).float()
        
        return embedding
    
class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, bidirectional, is_train):
        super(LSTMEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.is_train = is_train
        self.embedding_dropout = EmbeddingDropout()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first = True, bidirectional = bidirectional, num_layers= num_layers)
    
    def forward(self, factorized_data, factorized_data_lens, is_train):
        #factorized_data should be in cuda
        #factorized_data_lens should be in cpu
        if is_train:
            factorized_data_embedding = self.embedding_dropout(self.embedding, factorized_data, 0.75).to(device)
        else:
            factorized_data_embedding = self.embedding(factorized_data).to(device)
            
        packed_data = torch.nn.utils.rnn.pack_padded_sequence(factorized_data_embedding, factorized_data_lens, batch_first = True, enforce_sorted=False).float()
        
        packed_output_data, (h_n, c_n) = self.rnn(packed_data)
        output, output_lens = torch.nn.utils.rnn.pad_packed_sequence(packed_output_data, batch_first = True)
        
        return output, output_lens, h_n, c_n
    
class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers_enc, bidirectional_enc, is_train):
        super(LSTMDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers_enc = num_layers_enc 
        self.bidirectional_enc = bidirectional_enc
        self.is_train = is_train
        self.embedding_dropout = EmbeddingDropout()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0)
        self.lstm_cell_1 = nn.LSTMCell(embedding_dim + hidden_dim * num_layers_enc * (self.bidirectional_enc + 1), hidden_dim)
        self.lstm_cell_2 = nn.LSTMCell(hidden_dim, hidden_dim)
        self.final_prob = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, h_n, expanded_data, tf_prob, is_train):
        print(is_train)
        if is_train:
            expanded_data_embedding = self.embedding_dropout(self.embedding, expanded_data, 0.75).to(device)
        else:
            expanded_data_embedding = self.embedding(expanded_data).to(device)
        
        hidden_1 = torch.zeros(expanded_data_embedding.shape[0], self.hidden_dim).to(device)
        cell_state_1 = torch.zeros(expanded_data_embedding.shape[0], self.hidden_dim).to(device)
        hidden_2 = torch.zeros(expanded_data_embedding.shape[0], self.hidden_dim).to(device)
        cell_state_2 = torch.zeros(expanded_data_embedding.shape[0], self.hidden_dim).to(device)
        
        h_n = torch.reshape(h_n, (expanded_data_embedding.shape[0], -1))
        
        if is_train:
            print("Training")
            char_probs = []

            for i in range(expanded_data_embedding.shape[1]-1):
                rnn_input = torch.cat((expanded_data_embedding[:, i], h_n), dim=1)
                hidden_1, cell_state_1 = self.lstm_cell_1(rnn_input, (hidden_1, cell_state_1))
                hidden_2, cell_state_2 = self.lstm_cell_2(hidden_1, (hidden_2, cell_state_2))

                char_prob = self.final_prob(hidden_2)
                char_probs.append(char_prob)
            
                char_ind = torch.max(char_prob, dim=1)[1]
                tf_mask = torch.from_numpy(np.random.binomial(1, p = tf_prob, size = char_prob.shape[0])).to(device)

                for j in range(tf_mask.shape[0]):
                    if tf_mask[j] == 1:
                        expanded_data_embedding[j, i+1, :] = self.embedding(char_ind[j].view(1, 1)).to(device)
             
            return char_probs
    
        else:
            print("Testing")
            char_indices = []
            max_length = 29

            for i in range(max_length):
                rnn_input = torch.cat((expanded_data_embedding[:, 0, :], h_n), dim=1)
                hidden_1, cell_state_1 = self.lstm_cell_1(rnn_input, (hidden_1, cell_state_1))
                hidden_2, cell_state_2 = self.lstm_cell_2(hidden_1, (hidden_2, cell_state_2))

                char_prob = self.final_prob(hidden_2)

                char_ind = torch.max(char_prob, dim=1)[1]
                char_indices.append(char_ind)
                embedding = self.embedding(char_ind.view(char_ind.shape[0],1)).to(device)

            return char_indices

class Attention(nn.Module):
    def __init__(self, input_dim, key_dim, value_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.key = nn.Linear(input_dim, key_dim)
        self.value = nn.Linear(input_dim, value_dim)
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, enc_output, query, factorized_data_lens):
        key = self.key(enc_output)
        value = self.value(enc_output)
        
        # compute energy
        scale = 1.0/np.sqrt(self.key_dim)
        query = query.unsqueeze(1) # [B,Q] -> [B,1,Q]
        energy = torch.bmm(query, key.transpose(1,2))
        attn = energy.mul_(scale).squeeze(1)
        mask = torch.zeros([query.shape[0],key.shape[1]],dtype = torch.bool).to(device)
        for i in range(factorized_data_lens.shape[0]):
            mask[i,:factorized_data_lens[i]] = True
        attn = attn.masked_fill_(~mask,float('-inf'))
        attn = self.softmax(attn)
        
        # weight values
        context = torch.sum(value*attn.unsqueeze(2).repeat(1,1,value.size(2)),dim=1)
        
        return context
    
class LSTMDecoder_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers_enc, bidirectional_enc, is_train):
        super(LSTMDecoder_Attention, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers_enc = num_layers_enc 
        self.bidirectional_enc = bidirectional_enc
        self.is_train = is_train
        self.embedding_dropout = EmbeddingDropout()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0)
        self.lstm_cell_1 = nn.LSTMCell(embedding_dim + hidden_dim * num_layers_enc * (self.bidirectional_enc + 1), hidden_dim)
        self.lstm_cell_2 = nn.LSTMCell(hidden_dim, hidden_dim)
        self.attention = Attention(self.hidden_dim * 2, self.hidden_dim, self.hidden_dim)
        self.final_prob = nn.Linear(hidden_dim * 2, vocab_size)
        
    def forward(self, enc_output, h_n, expanded_data, factorized_data_lens, tf_prob, is_train):
        if is_train:
            expanded_data_embedding = self.embedding_dropout(self.embedding, expanded_data, 0.75).to(device)
        else:
            expanded_data_embedding = self.embedding(expanded_data).to(device)
        
        hidden_1 = torch.zeros(expanded_data_embedding.shape[0], self.hidden_dim).to(device)
        cell_state_1 = torch.zeros(expanded_data_embedding.shape[0], self.hidden_dim).to(device)
        hidden_2 = torch.zeros(expanded_data_embedding.shape[0], self.hidden_dim).to(device)
        cell_state_2 = torch.zeros(expanded_data_embedding.shape[0], self.hidden_dim).to(device)
        
        h_n = torch.reshape(h_n, (expanded_data_embedding.shape[0], -1))
        
        if is_train:
            print("Training")
            char_probs = []

            for i in range(expanded_data_embedding.shape[1]-1):
                rnn_input = torch.cat((expanded_data_embedding[:, i], h_n), dim=1)
                hidden_1, cell_state_1 = self.lstm_cell_1(rnn_input, (hidden_1, cell_state_1))
                hidden_2, cell_state_2 = self.lstm_cell_2(hidden_1, (hidden_2, cell_state_2))
                
                context = self.attention(enc_output, hidden_2, factorized_data_lens)
                
                char_prob = self.final_prob(torch.cat((hidden_2, context), dim=1))
                char_probs.append(char_prob)
            
                char_ind = torch.max(char_prob, dim=1)[1]
                tf_mask = torch.from_numpy(np.random.binomial(1, p = tf_prob, size = char_prob.shape[0])).to(device)

                for j in range(tf_mask.shape[0]):
                    if tf_mask[j] == 1:
                        expanded_data_embedding[j, i+1, :] = self.embedding(char_ind[j].view(1, 1)).to(device)
             
            return char_probs
    
        else:
            print("Testing")
            char_indices = []
            max_length = 29

            for i in range(max_length):
                rnn_input = torch.cat((expanded_data_embedding[:, 0, :], h_n), dim=1)
                hidden_1, cell_state_1 = self.lstm_cell_1(rnn_input, (hidden_1, cell_state_1))
                hidden_2, cell_state_2 = self.lstm_cell_2(hidden_1, (hidden_2, cell_state_2))
                
                context = self.attention(enc_output, hidden_2, factorized_data_lens)
                
                char_prob = self.final_prob(torch.cat((hidden_2, context), dim=1))

                char_ind = torch.max(char_prob, dim=1)[1]
                char_indices.append(char_ind)
                embedding = self.embedding(char_ind.view(char_ind.shape[0],1)).to(device)

            return char_indices
        
    def recognize_beam(self, enc_output, h_n, expanded_data, factorized_data_lens, mapping_dict):
        # search params
        beam = 10
        nbest = 1
        maxlen = 29

        hidden_1 = torch.zeros(expanded_data.shape[0], self.hidden_dim).to(device)
        cell_state_1 = torch.zeros(expanded_data.shape[0], self.hidden_dim).to(device)
        hidden_2 = torch.zeros(expanded_data.shape[0], self.hidden_dim).to(device)
        cell_state_2 = torch.zeros(expanded_data.shape[0], self.hidden_dim).to(device)
        
        h_n = torch.reshape(h_n, (expanded_data.shape[0], -1))
        
        y = mapping_dict['start']
        vy = hidden_1.new_zeros(1).long()

        hyp = {'score': 0.0, 'yseq': [y]}
        hyps = [hyp]
        ended_hyps = []

        for i in range(maxlen):
            hyps_best_kept = []
            for hyp in hyps:
                vy[0] = hyp['yseq'][i]
                embedded = self.embedding(vy)
                
                rnn_input = torch.cat((embedded, h_n), dim=1)
                hidden_1, cell_state_1 = self.lstm_cell_1(rnn_input, (hidden_1, cell_state_1))
                hidden_2, cell_state_2 = self.lstm_cell_2(hidden_1, (hidden_2, cell_state_2))

                context = self.attention(enc_output, hidden_2, factorized_data_lens)

                char_prob = self.final_prob(torch.cat((hidden_2, context), dim=1))
                local_scores = F.log_softmax(char_prob, dim=1)

                local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)

                for j in range(beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])
                    hyps_best_kept.append(new_hyp)

            hyps_best_kept = sorted(hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]
            hyps = hyps_best_kept

            if i == maxlen - 1:
                for hyp in hyps:
                    hyp['yseq'].append(mapping_dict['end'])

            remained_hyps = []

            for hyp in hyps:
                if hyp['yseq'][-1] == mapping_dict['end']:
                    ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            hyps = remained_hyps
            if len(hyps) > 0:
                pass
            else:
                break

        nbest_hyps = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), nbest)]

        return nbest_hyps

class F2E_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, bidirectional, is_train):
        super(F2E_LSTM, self).__init__()
        self.is_train = is_train
        self.Encoder = LSTMEncoder(vocab_size, embedding_dim, hidden_dim, num_layers, bidirectional, is_train)
        self.Decoder = LSTMDecoder(vocab_size, embedding_dim, hidden_dim, num_layers, bidirectional, is_train)

    def forward(self, factorized_data, factorized_data_lens, expanded_data, tf_prob):
        if self.is_train:
            output, output_lens, h_n, c_n = self.Encoder(factorized_data, factorized_data_lens, True)
            char_probs = self.Decoder(h_n, expanded_data, tf_prob, True)

            return char_probs
        else:
            output, output_lens, h_n, c_n = self.Encoder(factorized_data, factorized_data_lens, False)
            char_inds = self.Decoder(h_n, expanded_data, tf_prob, False)
            
            return char_inds
        
class F2E_LSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, bidirectional, is_train):
        super(F2E_LSTM_Attention, self).__init__()
        self.is_train = is_train
        self.Encoder = LSTMEncoder(vocab_size, embedding_dim, hidden_dim, num_layers, bidirectional, is_train)
        self.Decoder = LSTMDecoder_Attention(vocab_size, embedding_dim, hidden_dim, num_layers, bidirectional, is_train)

    def forward(self, factorized_data, factorized_data_lens, expanded_data, tf_prob, mapping_dict, decoding):
        if self.is_train:
            output, output_lens, h_n, c_n = self.Encoder(factorized_data, factorized_data_lens, True)
            char_probs = self.Decoder(output, h_n, expanded_data, factorized_data_lens, tf_prob, True)

            return char_probs
        else:
            if decoding == 'beam':
                output, output_lens, h_n, c_n = self.Encoder(factorized_data, factorized_data_lens, False)
                nbest_hyps = self.Decoder.recognize_beam(output, h_n, expanded_data, factorized_data_lens, mapping_dict)

                return nbest_hyps
            
            elif decoding == 'greedy':
                output, output_lens, h_n, c_n = self.Encoder(factorized_data, factorized_data_lens, False)
                char_inds = self.Decoder(output, h_n, expanded_data, factorized_data_lens, 0, False)

                return char_inds