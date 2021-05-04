import torch
from torch.utils.data import Dataset

class Factorized2ExpandedDataset(Dataset):
    '''
    Dataset class for conversion of a factorized polynomial to its expanded form.
    '''
    def __init__(self, factorized_form, expanded_form):
        self.factorized_form = factorized_form
        self.expanded_form = expanded_form
        
    def __len__(self):
        return len(self.factorized_form)

    def __getitem__(self, index):
        return torch.tensor(self.factorized_form[index]), torch.tensor(self.expanded_form[index])
    
def collate(batch_data):
    '''
    Returns padded factorized and expanded data, and length of unpadded factorized and expanded forms
    '''
    factorized_data = []
    expanded_data = []

    for i in range(len(batch_data)):
        factorized_form , expanded_form = batch_data[i]
        factorized_data.append(factorized_form.long())
        expanded_data.append(expanded_form.long())

    factorized_data_lens = torch.LongTensor([len(seq) for seq in factorized_data])
    expanded_data_lens = torch.LongTensor([len(seq) for seq in expanded_data])

    factorized_data = torch.nn.utils.rnn.pad_sequence(factorized_data, batch_first = True)
    expanded_data = torch.nn.utils.rnn.pad_sequence(expanded_data, batch_first = True)

    return factorized_data, factorized_data_lens, expanded_data, expanded_data_lens