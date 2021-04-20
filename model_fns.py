import time
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.optim as optim

# define test function 
def test(model: nn.Module, 
         test_loader: DataLoader, 
         criterion, 
         device: str):
    test_loss = 0
    acc_store = preci_store = recall_store = f1_store = 0
    
    for images, labels in test_loader:
        # labels = extract_labels(labels, pos_label, device)
        images, labels = images.to(device), labels.to(device)
        
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        
        # output = torch.exp(output)
        output = output.argmax(1)
        # output = F.one_hot(output, num_classes=2)
        # true_pos, true_neg, false_pos, false_neg = get_confusion_matrix(labels, output)
        # accuracy, precision, recall, f1 = get_metrics(true_pos, true_neg, false_pos, false_neg)
        # acc_store += accuracy
        # preci_store += precision
        # recall_store += recall
        # f1_store += f1

    test_loss /= len(test_loader)
    acc_store /= len(test_loader)
    preci_store /= len(test_loader)
    recall_store /= len(test_loader)
    f1_store /= len(test_loader)

    return test_loss, acc_store, preci_store, recall_store, f1_store

# define train function
def train(model: nn.Module, 
          train_loader: DataLoader, 
          test_loader: DataLoader, 
          no_of_epochs: int, 
        #   pos_label: str, 
        #   weight: torch.tensor, 
        #   pos_weight: torch.tensor, 
          save_dir:str=None, 
          patience:int=10, 
          device:str="cuda", 
          lr_scheduler:bool=False):
    model.to(device)

    if save_dir != None:
        os.mkdir(save_dir)

    criterion = nn.BCEWithLogitsLoss()#weight= weight, pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience//3, verbose=True)

    criterion.to(device)

    running_loss = 0

    train_loss_store, test_loss_store = [], []
    acc_store, preci_store, recall_store, f1_store = [], [], [], []
    # early_stopper = EarlyStopping(patience=patience)

    start = time.time()
    for epoch in range(1, no_of_epochs+1):
        # train mode for training
        model.train()
        count = 0
        for images, labels in train_loader:
            try:
                # labels = extract_labels(labels, pos_label, device)
                images, labels = images.to(device), labels.to(device)
                images = torch.squeeze(images)

                optimizer.zero_grad()
                
                output = model.forward(images)
                output = torch.squeeze(output, 0)

                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # print(count)
                count += 1
            except Exception as e:
                print(count)
                print(images.shape)
                print(f"loss: {running_loss/count+1}")
                raise e

        # eval mode for predictions
        model.eval()

        # turn off gradients for test
        with torch.no_grad():
            test_loss, accuracy, precision, recall, f1 = test(model, test_loader, criterion, device)

        train_loss_store.append(running_loss/len(train_loader))
        test_loss_store.append(test_loss)
        acc_store.append(accuracy)
        preci_store.append(precision)
        recall_store.append(recall)
        f1_store.append(f1)

        print(f"Epoch: {epoch}/{no_of_epochs} - ",
              f"Training Loss: {train_loss_store[-1]:.3f} - ",
              f"Test Loss: {test_loss_store[-1]:.3f} - ",
              f"Test Accuracy: {acc_store[-1]:.3f} - ", 
              f"Test Precision: {preci_store[-1]:.3f} - ", 
              f"Test Recall: {recall_store[-1]:.3f} - ", 
              f"Test F1 Score: {f1_store[-1]:.3f}")

        if save_dir != None:
            torch.save(model.state_dict(), f"{save_dir}/{epoch}.pt")
        
        # if early_stopper.stop(test_loss_store[-1]):
        #     print("Model has overfit, early stopping...")
        #     break

        if lr_scheduler:
            scheduler.step(test_loss_store[-1])
        
        running_loss = 0

    print(f"Run time: {(time.time() - start)/60:.3f} min")
    
    return train_loss_store, test_loss_store, acc_store, preci_store, recall_store, f1_store
