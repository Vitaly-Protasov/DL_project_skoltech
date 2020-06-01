import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset


def run_epoch(model, optimizer, criterion, dataloader, epoch, mode='train'):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if mode=='train':
        model.train()
    else:
        model.eval()
    model.to(DEVICE)
    epoch_loss = 0.0
    accuracy = 0
    dataloader
    
    try:
        dataloader._form_tensors()
    except:
        raise RuntimeError('You use a weird type of dataset. It shoulb be DatasetBuilder.')

    for i, (starts, contexts, ends, labels) in enumerate(dataloader._form_tensors()):
        starts, contexts, ends = starts.to(DEVICE), contexts.to(DEVICE), ends.to(DEVICE)
        labels = labels.to(DEVICE)

        iteration_n = epoch * i + i
        
        code_vector, y_pred = model(starts, contexts, ends)
        loss = criterion(y_pred.type(torch.float), labels.type(torch.float))

        y_pred_classes = torch.argmax(y_pred, dim=1)
        y_true_classes = torch.argmax(labels, dim=1)
        accuracy += (y_pred_classes == y_true_classes).float().mean()
        
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss += loss.item()
        
    
    return epoch_loss / len(dataloader), accuracy / len(dataloader) * 100
    
def train(model, optimizer, criterion, train_loader, n_epochs,
          scheduler=None, checkpoint=True, freq=None, verbose=True):
    if verbose and freq is None:
        freq = max(1, n_epochs // 10)
    
    best_val_loss = float('+inf')
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []
    for epoch in range(n_epochs):
        train_loss, train_accuracy = run_epoch(model, optimizer, criterion, train_loader, epoch, 'train')
        val_loss, val_accuracy = run_epoch(model, None, criterion, val_loader, epoch, 'val')

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_accuracy)
        val_acc_list.append(val_accuracy)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
        if checkpoint:
            torch.save(model.state_dict(), './model_best.pth')
        
        if scheduler is not None:
            scheduler.step(val_loss)
        if verbose and epoch % freq == 0:
            print("Epoch {}: train loss - {} | validation loss - {}".format(epoch, train_loss, val_loss))
            print(f'Epoch {epoch}: f1_val={f1_val}, rec_val={rec_val}, prec_val={prec_val}')
        
    return train_loss_list, train_acc_list