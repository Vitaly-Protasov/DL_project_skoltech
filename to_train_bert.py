import create_vocab
import data_to_tensors
import model_implementation
from train_class import TrainingModule

import torch
import torch.nn as nn
import torch.nn.functional as F
import random 
from torch.utils.data import DataLoader

try:
    import transformers
except:
    raise RuntimeError('Please: pip install transformers')

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
                                                        

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)  

    bert = True
    bert_params = dict()
    bert_params['num_attention_heads'] = 1
    bert_params['num_transformer_layers'] = 1
    bert_params['intermediate_size'] = 32
        
    model = model_implementation.code2vec_model(values_vocab_size = len(word2idx), 
                                 paths_vocab_size = len(path2idx), 
                                 labels_num = len(target2idx), bert=bert, bert_params=bert_params)
    ########################################################################################
    N_EPOCHS = 20
    LR = 3e-3
    WD = 0.8e-5

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.CrossEntropyLoss()

    train_class = TrainingModule(model, optimizer, criterion, train_loader, val_loader, N_EPOCHS, idx2target)
    list_train_loss, list_val_loss, list_train_precision, list_val_precision,list_train_recall, list_val_recall, list_train_f1, list_val_f1 = train_class.train()

        
if __name__== "__main__":
  main()
