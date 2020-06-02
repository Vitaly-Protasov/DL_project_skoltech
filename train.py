import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset


def run_epoch(model, optimizer, criterion, dataloader, epoch, device, mode='train'):
    
    if mode=='train':
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    accuracy = 0
    
    try:
        dataloader._form_tensors()
    except:
        raise RuntimeError('You use a weird type of dataset. It shoulb be DatasetBuilder.')

    for i, (starts, contexts, ends, labels) in enumerate(dataloader._form_tensors()):
        starts, contexts, ends = starts.to(device), contexts.to(device), ends.to(device)
        labels = labels.to(device)
        
        code_vector, y_pred = model(starts, contexts, ends)
        loss = criterion(y_pred, torch.argmax(labels, dim = 1))
        
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss += loss.item()

        break
    
    return epoch_loss / len(dataloader)#, accuracy / len(dataloader) * 100
    
def train(model, optimizer, criterion, train_loader, val_loader, n_epochs,
          scheduler=None, checkpoint=True, freq=None, verbose=True):
    if verbose and freq is None:
        freq = max(1, n_epochs // 10)
    
    best_val_loss = float('+inf')
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = model.to(DEVICE)
    criterion = criterion.to(DEVICE)
    for epoch in range(n_epochs):
        train_loss = run_epoch(model, optimizer, criterion, train_loader, epoch, device = DEVICE, mode = 'train')
        try:
          val_loss = run_epoch(model, None, criterion, val_loader, epoch, device = DEVICE, mode = 'val')
        except:
          val_loss = -1 
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        # train_acc_list.append(train_accuracy)
        # val_acc_list.append(val_accuracy)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            if checkpoint:
                torch.save(model.state_dict(), './model_best.pth')
        
        if scheduler is not None:
            scheduler.step(val_loss)
        if verbose and epoch % freq == 0:
            print("Epoch {}: train loss - {} | validation loss - {}".format(epoch, train_loss, val_loss))

            # print(f'Epoch {epoch}: f1_val={f1_val}, rec_val={rec_val}, prec_val={prec_val}')
        
    return train_loss_list, train_acc_list