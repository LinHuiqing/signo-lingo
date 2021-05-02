import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report


class EarlyStopping:
    def __init__(self, 
                 patience:int=10, 
                 delta:int=0, 
                 logger=None) -> None:
        self.patience = patience
        self.delta = delta
        self.logger = logger

        self.best_score = float("inf")
        self.overfit_count = 0

    def stop(self, loss):
        threshold = self.best_score + self.delta
        if loss > threshold:
            self.overfit_count += 1
            print_msg = f"Increment early stopper to {self.overfit_count} because val loss ({loss}) is greater than threshold ({threshold})"
            if self.logger:
                self.logger.info(print_msg)
            else:
                print(print_msg)
        else:
            self.overfit_count = 0
        
        self.best_score = min(self.best_score, loss)
        
        if self.overfit_count >= self.patience:
            return True
        else:
            return False

def _train_epoch(model, criterion, optimizer, dataloader, device):
    model.train()
    losses = []
    all_label = []
    all_pred = []
    
    with tqdm(dataloader, unit="batch") as tepoch:
        for inputs, labels in tepoch:
            # get the inputs and labels
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            # outputs = torch.squeeze(outputs, dim=0)
            if isinstance(outputs, list):
                outputs = outputs[0]

            # compute the loss
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            # compute the accuracy
            prediction = torch.max(outputs, 1)[1]
            all_label.extend(labels)
            all_pred.extend(prediction)
            # print(labels.cpu().data.squeeze().numpy())
            # print(labels.cpu().data.numpy())
            score = accuracy_score(labels.cpu().data.numpy(), prediction.cpu().data.numpy())
            
            # backward & optimize
            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item(), accuracy=score)

    # Compute the average loss & accuracy
    train_loss = sum(losses)/len(losses)
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    train_acc = accuracy_score(all_label.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())

    return train_loss, train_acc

def _val_epoch(model, criterion, dataloader, device):
    model.eval()
    losses = []
    all_label = []
    all_pred = []

    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            for inputs, labels in tepoch:
                # get the inputs and labels
                inputs, labels = inputs.to(device), labels.to(device)
                # forward
                outputs = model(inputs)
                if isinstance(outputs, list):
                    outputs = outputs[0]
                # compute the loss
                loss = criterion(outputs, labels)
                losses.append(loss.item())
                # collect labels & prediction
                prediction = torch.max(outputs, 1)[1]
                all_label.extend(labels)
                all_pred.extend(prediction)

                score = accuracy_score(labels.cpu().data.numpy(), prediction.cpu().data.numpy())

                tepoch.set_postfix(loss=loss.item(), accuracy=score)
                
    # Compute the average loss & accuracy
    val_loss = sum(losses)/len(losses)
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    val_acc = accuracy_score(all_label.cpu().data.numpy(), all_pred.cpu().data.numpy())
    
    return val_loss, val_acc

def train(model: nn.Module, 
          train_loader: DataLoader, 
          val_loader: DataLoader, 
          no_of_epochs:int, 
          logger,
          writer,
          save_dir:str=None, 
          save_checkpoint:bool=False,
          load_dir:str=None,
          load_epoch:int=None,
          load_checkpoint:bool=False,
          device:str="cuda", 
          patience:int=10, 
          optimizer_lr:int=0.001, 
          weight_decay:int=0, 
          use_scheduler:bool=False):
    
    if save_dir and not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=optimizer_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience//3, verbose=True)

    start_epoch = 1

    if load_checkpoint and load_dir and load_epoch:
        checkpoint = torch.load(f"{load_dir}/{load_epoch}-checkpoint.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # Start training
    logger.info("Training Started".center(60, '#'))

    early_stopper = EarlyStopping(patience=patience, logger=logger)

    for epoch in range(start_epoch, no_of_epochs+1):
        logger.info(f"Epoch {epoch}")
        
        # Train the model
        train_loss, train_acc = _train_epoch(model, criterion, optimizer, train_loader, device)

        writer.add_scalars('Loss', {'train': train_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc}, epoch)
        logger.info("Average Training Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(epoch, train_loss, train_acc*100))

        # Validate the model
        val_loss, val_acc = _val_epoch(model, criterion, val_loader, device)

        writer.add_scalars('Loss', {'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'val': val_acc}, epoch)
        logger.info("Average Validation Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(epoch, val_loss, val_acc*100))

        if save_dir:
            if save_checkpoint:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                }, f"{save_dir}/{epoch}-checkpoint.pt")
            else:
                torch.save(model.state_dict(), f"{save_dir}/{epoch}.pt")

        # Save model
        logger.info(f"Epoch {epoch} Model Saved".center(60, '#'))

        if early_stopper.stop(val_loss):
            logger.info("Model has overfit, early stopping...")
            break
        
        if use_scheduler:
            scheduler.step(val_loss)

    logger.info("Training Finished".center(60, '#'))

    return model
