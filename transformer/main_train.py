from dataprocessing import *
from dataloader import *
from model import *
from train_test import *
from torch.utils.data import DataLoader
import pickle

try:
  mapping_dict = pickle.load(open("mapping_dict.pkl","rb"))
except:
  file = open('dataset.txt','r')
  data = file.read()
  data = data.split('\n')[:-1]
  data_ = list(map(lambda x: x.replace('**','^'), data))

  inp_opt_chars = create_inp_opt(data_)
  vocab = create_vocab(inp_opt_chars)
  mapping_dict = create_mapping(list(vocab))

  pickle.dump(mapping_dict, open("mapping_dict.pkl", "wb"))

train_file = open('train.txt','r')
train_data = train_file.read()
train_data = train_data.split('\n')[:-1]
train_data_ = list(map(lambda x: x.replace('**','^'), train_data))

train_inp_opt_chars = create_inp_opt(train_data_)
X_train, y_train = inp_opt(train_inp_opt_chars, mapping_dict)

train_dataset = Factorized2ExpandedDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn= collate)

SRC_VOCAB_SIZE = len(mapping_dict)
TGT_VOCAB_SIZE = len(mapping_dict)
EMB_SIZE = 256
NHEAD = 8
FFN_HID_DIM = 1024
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 2
NUM_EPOCHS = 10

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                                 EMB_SIZE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                                 FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(device)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=mapping_dict['pad'])

optimizer = torch.optim.Adam(
    transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(transformer):,} trainable parameters')

for i in range(NUM_EPOCHS):
    print(train_epoch(transformer, train_loader, optimizer))
    
torch.save(transformer, "model_transformer.pt")
    
