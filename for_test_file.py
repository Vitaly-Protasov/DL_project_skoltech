import create_vocab
import data_to_tensors
import model_implementation
from train import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import random 

from torch.utils.data import *

def main():
    SEED = 1337
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    


    dict_path = 'data/java-small/java-small.dict.c2v'
    word2idx, path2idx, target2idx, idx2target = create_vocab.create_vocab(dict_path)

    path_for_train = 'data/java-small/java-small.train.c2v'
    train_dataset = data_to_tensors.TextDataset(path_for_train, 
                                                        word2idx, 
                                                        path2idx, 
                                                        target2idx)

    path_for_val = 'data/java-small/java-small.val.c2v'
    val_dataset = data_to_tensors.TextDataset(path_for_val, 
                                                        word2idx, 
                                                        path2idx, 
                                                        target2idx)
                                                        
    path_for_test = 'data/java-small/java-small.test.c2v'
    test_dataset = data_to_tensors.TextDataset(path_for_test, 
                                                        word2idx, 
                                                        path2idx, 
                                                        target2idx)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)                                 
        
    model = model_implementation.code2vec_model(values_vocab_size = len(word2idx), 
                             paths_vocab_size = len(path2idx), 
                             labels_num = len(target2idx))
    ########################################################################################
    N_EPOCHS = 50
    LR = 3e-3

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    early_stop = False # ставите True и тогда будет обучение ток для одного батча
    list_train_loss, list_val_loss, list_train_precision, list_val_precision,list_train_recall, list_val_recall, list_train_f1, list_val_f1 = train(model = model, optimizer = optimizer,
                                                                                                                                                    criterion = criterion, train_loader = train_loader,
                                                                                                                                                    val_loader = val_loader, test_loader = test_loader,
                                                                                                                                                    epochs = N_EPOCHS, idx2target_vocab = idx2target, 
                                                                                                                                                    scheduler=None, checkpoint=True, early_stop = early_stop)

    
if __name__== "__main__":
  main()
