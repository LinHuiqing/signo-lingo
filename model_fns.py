import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report


# define val function 
def val(model: nn.Module, 
         val_loader: DataLoader, 
         criterion, 
         device: str):
    val_loss = 0
    val_acc = 0 
    metrics_store = {}
    
    with tqdm(val_loader, unit="batch") as tepoch:
        for images, labels in tepoch:
            images, labels = images.to(device), labels.to(device)
            labels_loss = labels.long()
            labels_loss = labels.argmax(dim=1)
            
            output = model.forward(images)
            loss = criterion(output, labels_loss).item()
            val_loss += loss
            
            output = output.argmax(1)
            output = F.one_hot(output, num_classes=labels.shape[1])

            output = output.cpu()
            labels = labels.cpu()

            accuracy = accuracy_score(labels, output)
            metrics_report = classification_report(labels, output, digits=3, output_dict=True, zero_division=0)

            val_acc += accuracy
            for label, metrics_dict in metrics_report.items():
                metrics_store[label] = metrics_store.get(label, {})
                for metric_type, metric_val in metrics_dict.items():
                    metrics_store[label][metric_type] = metrics_store[label].get(metric_type, 0)
                    metrics_store[label][metric_type] += metric_val

            tepoch.set_postfix(loss=loss, accuracy=accuracy)

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    for label, metrics_dict in metrics_report.items():
        for metric_type, metric_val in metrics_dict.items():
            metrics_store[label][metric_type] /= len(val_loader)

    return val_loss, val_acc, metrics_store

# define train function
def train(model: nn.Module, 
          train_loader: DataLoader, 
          val_loader: DataLoader, 
          no_of_epochs: int, 
          save_dir:str=None, 
          save_checkpoint:bool=False,
          load_dir:str=None,
          load_epoch:int=None,
          load_checkpoint:bool=False,
          patience:int=10, 
          device:str="cuda", 
          lr_scheduler:bool=False):
    model.to(device)

    if save_dir != None and not os.path.exists(save_dir):
        os.mkdir(save_dir)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience//3, verbose=True)

    criterion.to(device)

    running_loss = 0
    running_acc = 0

    train_loss_store, train_acc_store = [], []
    val_loss_store, val_acc_store = [], []
    val_metrics_store = []
    # early_stopper = EarlyStopping(patience=patience)
    start_epoch = 1

    if load_checkpoint and load_dir and load_epoch:
        checkpoint = torch.load(f"{load_dir}/{load_epoch}-checkpoint.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']

    start = time.time()
    for epoch in range(start_epoch, no_of_epochs+1):
        # train mode for training
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)
                labels = labels.long()
                labels = labels.argmax(dim=1)
                optimizer.zero_grad()
                
                output = model.forward(images)

                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                correct = (output.argmax(1) == labels).sum().item()
                train_acc = correct / len(images)

                running_acc += train_acc

                tepoch.set_postfix(loss=loss.item(), accuracy=train_acc)

        # eval mode for predictions
        model.eval()

        # turn off gradients for val
        with torch.no_grad():
            val_loss, val_acc, val_metrics = val(model, val_loader, criterion, device)

        train_loss_store.append(running_loss/len(train_loader))
        train_acc_store.append(running_acc/len(train_loader))
        val_loss_store.append(val_loss)
        val_acc_store.append(val_acc)
        val_metrics_store.append(val_metrics)

        print(f"Epoch: {epoch}/{no_of_epochs} - ",
              f"Training Loss: {train_loss_store[-1]:.3f} - ",
              f"Training Accuracy: {train_acc_store[-1]:.3f} - ",
              f"Val Loss: {val_loss_store[-1]:.3f} - ",
              f"Val Accuracy: {val_acc_store[-1]:.3f} - ")

        if save_dir != None:
            if save_checkpoint:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss,
                }, f"{save_dir}/{epoch}-checkpoint.pt")
            else:
                torch.save(model.state_dict(), f"{save_dir}/{epoch}.pt")
        
        # if early_stopper.stop(val_loss_store[-1]):
        #     print("Model has overfit, early stopping...")
        #     break

        if lr_scheduler:
            scheduler.step(val_loss_store[-1])
        
        running_loss = 0
        running_acc = 0

    print(f"Run time: {(time.time() - start)/60:.3f} min")
    
    return train_loss_store, train_acc_store, val_loss_store, val_acc_store, val_metrics_store #, preci_store, recall_store, f1_store
