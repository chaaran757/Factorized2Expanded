from data_processing import *
from data_loader import *
from model import *
from train_test import *
from torch.utils.data import DataLoader

#Data
file = open('dataset.txt','r')
data = file.read()
data = data.split('\n')[:-1]
data_ = list(map(lambda x: x.replace('**','^'), data))

inp_opt_chars = create_inp_opt(data_)
vocab = create_vocab(inp_opt_chars)
mapping_dict = create_mapping(list(vocab))

test_file = open('test.txt','r')
test_data = test_file.read()
test_data = test_data.split('\n')[:-1]
test_data_ = list(map(lambda x: x.replace('**','^'), test_data))

test_inp_opt_chars = create_inp_opt(test_data_)
X_test, y_test = inp_opt(test_inp_opt_chars, mapping_dict)

test_dataset = Factorized2ExpandedDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn= collate)

#Model
model = F2E_LSTM_Attention(len(mapping_dict)+1, 128, 256, 2, True, False)
model = torch.load("model_lstm_attention.pt")
model.is_train = False

#Training
criterion = nn.CrossEntropyLoss(reduction = 'none')

accuracy, pred_gt_list = test_beam(model, test_loader, mapping_dict) #Greedy