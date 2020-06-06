import create_vocab
import data_to_tensors
import model_implementation
from train_class import TrainingModule

import torch
import torch.nn as nn
import torch.nn.functional as F
import random 
from torch.utils.data import DataLoader

def main(batch_size, lr, wd):
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)   
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)      
        
    model = model_implementation.code2vec_model(values_vocab_size = len(word2idx), 
                             paths_vocab_size = len(path2idx), 
                             labels_num = len(target2idx))
    ########################################################################################
    N_EPOCHS = 40
    LR = lr

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    train_class = TrainingModule(model, optimizer, criterion, train_loader, val_loader, test_loader, N_EPOCHS, idx2target)
    _, _, _, _,_, _, _, _ = train_class.train()

    
if __name__== "__main__":
  batch_size = int(input('Input batch size: '))
  lr = float(input('Input learning rate: '))
  wd = float(input('Input weight decay: '))

  main(batch_size, lr, wd)
