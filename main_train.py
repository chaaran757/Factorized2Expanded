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

train_file = open('train.txt','r')
train_data = train_file.read()
train_data = train_data.split('\n')[:-1]
train_data_ = list(map(lambda x: x.replace('**','^'), train_data))

train_inp_opt_chars = create_inp_opt(train_data_)
X_train, y_train = inp_opt(train_inp_opt_chars, mapping_dict)

train_dataset = Factorized2ExpandedDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn= collate)

#Model
model = F2E_LSTM_Attention(len(mapping_dict)+1, 128, 256, 2, True, True)

#Training
criterion = nn.CrossEntropyLoss(reduction = 'none')
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

num_epochs = 10
tf_prob = 0.10

for i in range(num_epochs):
    train(model, train_loader, criterion, optimizer, tf_prob, mapping_dict)
    torch.save(model, "model_lstm_attention.pt")
    
    if tf_prob < 0.40:
        tf_prob += 0.03
    
    if (i+1)%3 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/2
            
    if (i+1)%5 == 0:
        model.is_train = False
        print(test(model, test_loader, mapping_dict)) #Greedy
        model.is_train = True