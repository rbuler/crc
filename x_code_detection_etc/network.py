import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, confusion_matrix


class MILNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MILNetwork, self).__init__()
        
        self.instance_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU())
        
        self.aggregation = lambda x: torch.max(x, dim=1)[0]
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, bag):
        b, n, feats = bag.size()  # batch size, number of instances, number of features
        bag = bag.view(b * n, feats)
        x = self.instance_encoder(bag)
        x = x.view(b, n, -1)
        x = self.aggregation(x)
        x = self.classifier(x)
        return x


def train_net(model, criterion, optimizer, train_loader, valid_loader, num_epochs=10, patience=10):
    history = {'epoch': [], 'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        corrects = 0
        epoch_loss = 0
        model.train()
        for batch in train_loader:
            for bag in batch:
                optimizer.zero_grad()
                instances = torch.stack(bag['instances']).unsqueeze(0)
                bag_label = torch.tensor(bag['bag_label'], dtype=torch.float32)

                output = model(instances)
                loss = criterion(output[0], bag_label)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                preds = (output > 0.5).float()
                corrects += (preds == bag_label.item()).sum().item()

        train_loss = epoch_loss / len(train_loader)
        train_acc = corrects / len(train_loader)

        # val loop
        model.eval()
        val_corrects = 0
        val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                for bag in batch:
                    instances = torch.stack(bag['instances']).unsqueeze(0)
                    bag_label = torch.tensor(bag['bag_label'], dtype=torch.float32)

                    output = model(instances)
                    loss = criterion(output[0], bag_label)

                    val_loss += loss.item()
                    preds = (output > 0.5).float()
                    val_corrects += (preds == bag_label.item()).sum().item()
        
        val_loss /= len(valid_loader)
        val_acc = val_corrects / len(valid_loader)

        history['epoch'].append(epoch + 1)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # store state dict
            # TODO
            # return best model
            # best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return model, history


def test_net(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            for bag in batch:
                instances = torch.stack(bag['instances']).unsqueeze(0)
                bag_label = bag['bag_label']

                output = model(instances)
                preds = (output > 0.5).float().item()
                
                all_preds.append(preds)
                all_labels.append(bag_label.item())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1_score": f1_score(all_labels, all_preds, zero_division=0),
    }

    conf_matrix = confusion_matrix(all_labels, all_preds)


    return metrics, conf_matrix