from tqdm import tqdm
import numpy as np
import torch

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

def train(model, train_loader, criterion, optimizer, tf_prob, mapping_dict):
    model.train()
    model.to(device)
    
    progress_bar = enumerate(tqdm(train_loader))
    
    for i,(factorized_data, factorized_data_lens, expanded_data, expanded_data_lens) in progress_bar:
        with torch.autograd.set_detect_anomaly(True):
            factorized_data = factorized_data.to(device)
            expanded_data = expanded_data.to(device)
    
            char_probs = model(factorized_data, factorized_data_lens, expanded_data, tf_prob, mapping_dict, 'greedy')
            char_probs = torch.stack(char_probs, dim=2)
            
            mask = torch.zeros([len(expanded_data_lens), expanded_data_lens.max()-1],dtype = torch.bool)
            
            for i in range(len(expanded_data_lens)):
                mask[i, :expanded_data_lens[i]-1] = True
                
            mask = mask.to(device)
            loss = criterion(char_probs, expanded_data[:,1:].long())
            masked_loss = loss.masked_fill_(~mask,0).sum()
            
            print(masked_loss)
            
            masked_loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), 2)

            optimizer.step()
            
def test_beam(model, test_loader, mapping_dict):
    with torch.no_grad():
        model.eval()
        model.is_train = False
        model.to(device)
        
        pred_gt_list = []
        
        correct = 0
        total = 0
        
        progress_bar = enumerate(tqdm(test_loader))
        
        mapping_dict_inverse = {}
        
        for key, value in mapping_dict.items():
            mapping_dict_inverse[str(value)] = key
            
        for batch_num,(factorized_data, factorized_data_lens, expanded_data, expanded_data_lens) in progress_bar:
            factorized_data = factorized_data.to(device)
            expanded_data = expanded_data.to(device)

            nbest_hyps = model(factorized_data, factorized_data_lens, expanded_data, 0, mapping_dict, 'beam')
            
            transcript = []
            
            for i in range(1,len(nbest_hyps[0]['yseq'])):
                if nbest_hyps[0]['yseq'][i]  != mapping_dict['end']:
                    transcript.append(nbest_hyps[0]['yseq'][i])

            for i in range(len(transcript)):
                transcript[i] = mapping_dict_inverse[str(transcript[i])]
                
            ground_truth = ''.join(mapping_dict_inverse[str(expanded_data[0][i].cpu().numpy())] for i in range(1, expanded_data_lens[0]-1))
            predicted = ''.join(transcript[i] for i in range(len(transcript)))
                
            if batch_num == 0:
                print(ground_truth, predicted)

            total += 1

            if predicted == ground_truth:
                correct += 1
            
            pred_gt_list.append((ground_truth, predicted))
            
        return correct/total, pred_gt_list
    
def test(model, test_loader, mapping_dict):
    with torch.no_grad():
        model.eval()
        model.is_train = False
        model.to(device)
        
        pred_gt_list = []

        correct = 0
        total = 0
        
        progress_bar = enumerate(tqdm(test_loader))
        
        mapping_dict_inverse = {}
        
        for key, value in mapping_dict.items():
            mapping_dict_inverse[str(value)] = key
            
        for batch_num,(factorized_data, factorized_data_lens, expanded_data, expanded_data_lens) in progress_bar:
            factorized_data = factorized_data.to(device)
            expanded_data = expanded_data.to(device)

            char_indices = model(factorized_data, factorized_data_lens, expanded_data, 0, mapping_dict, 'greedy')

            char_indices = torch.stack(char_indices, dim=1)
            
            print(char_indices.shape)
            
            for j in range(char_indices.shape[0]):
                transcript = []
                for i in range(char_indices[j].shape[0]):
                    if char_indices[j][i] != mapping_dict['end'] :    
                        transcript.append(char_indices[j][i].cpu().numpy())
                
                for i in range(len(transcript)):
                    transcript[i] = mapping_dict_inverse[str(transcript[i])]
                
                ground_truth = ''.join(mapping_dict_inverse[str(expanded_data[j][i].cpu().numpy())] for i in range(1, expanded_data_lens[j]-1))
                predicted = ''.join(transcript[i] for i in range(len(transcript)))
                
                if batch_num == 0:
                    print(ground_truth, predicted)

                total += 1

                if predicted == ground_truth:
                    correct += 1
                
                pred_gt_list.append((ground_truth, predicted))
                
        return correct/total, pred_gt_list