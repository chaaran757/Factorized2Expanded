import numpy as np
import itertools
from sklearn.model_selection import train_test_split

def create_inp_opt(data):
    inp_opt = map(lambda x:x.split('='), data)
    inp_opt_chars = list(map(lambda x: [list(x[0]), list(x[1])], inp_opt))
    
    return inp_opt_chars

def create_vocab(data):
    inp_opt_merged = list(map(lambda x: list(itertools.chain(*x)), data))
    vocab = set(itertools.chain(*inp_opt_merged))
    vocab = vocab.union({'start', 'end'})
    
    return vocab

def create_mapping(vocab):
    mapping = {}
    for i in range(len(vocab)):
        mapping[vocab[i]] = i+1
    
    mapping['pad'] = 0
    
    return mapping

def inp_opt(inp_opt_chars, mapping):
    inp_opt_mapped = list(map(lambda x:[[mapping['start']]+[mapping[i] for i in x[0]]+[mapping['end']], 
                                  [mapping['start']]+[mapping[i] for i in x[1]]+[mapping['end']]], inp_opt_chars))
    
    inp = list(map(lambda x:x[0], inp_opt_mapped))
    opt = list(map(lambda x:x[1], inp_opt_mapped))
    
    return inp, opt

file = open('dataset.txt','r')
data = file.read()

data = data.split('\n')[:-1]

train, test = train_test_split(data, test_size=0.10, random_state=1)
train, val = train_test_split(train, test_size=0.10, random_state=1)

with open('train.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % exp for exp in train)
    
with open('valid.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % exp for exp in val)
    
with open('test.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % exp for exp in test)