from dataprocessing import *
from dataloader import *
from model import *
from train_test import *
import pickle

mapping_dict = pickle.load(open("mapping_dict.pkl","rb"))

inv_mapping_dict = {}

for key, value in mapping_dict.items():
    inv_mapping_dict[value] = key
    
test_file = open('test.txt','r')
test_data = test_file.read()
test_data = test_data.split('\n')[:-1]
test_data_ = list(map(lambda x: x.replace('**','^'), test_data))

test_inp_opt_chars = create_inp_opt(test_data_)
X_test, y_test = inp_opt(test_inp_opt_chars, mapping_dict)

correct = 0

for i in range(len(X_test)):
    src = X_test[i]
    pred = expand(transformer, src, mapping_dict, inv_mapping_dict)
    gt = " ".join([inv_mapping_dict[tok] for tok in y_test[i]]).replace("start", "").replace("end", "")
    
    if pred == gt:
        correct += 1
        
print("Accuracy: ", correct/len(X_test))
