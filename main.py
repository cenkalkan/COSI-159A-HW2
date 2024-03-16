# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

import time

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

from model import SphereCNN, AngularPenaltySMLoss
from dataloader import LFW4Training, LFW4Eval
from parser import parse_args
from utils import set_seed, AverageMeter
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F


def eval(data_loader: DataLoader, model: SphereCNN, device: torch.device):
    model.eval()
    model.feature = True
    sim_func = nn.CosineSimilarity()
    labels, scores = [], []

    with torch.no_grad():
        for img_1, img_2, label in data_loader:
            img_1, img_2, label = img_1.to(device), img_2.to(device), label.to(device)

            feat_1 = model(img_1, None)
            feat_2 = model(img_2, None)

            sim = sim_func(feat_1, feat_2).cpu().numpy()
            scores.extend(sim)
            labels.extend(label.cpu().numpy())

    # Convert similarities and labels to arrays
    scores = np.array(scores)
    labels = np.array(labels)

    # Compute ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Find the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Convert scores to binary predictions based on the optimal threshold
    predictions = (scores > optimal_threshold).astype(int)

    # Calculate the accuracy
    accuracy = np.mean(predictions == labels)

    print(f'ROC AUC: {roc_auc:.4f}, Optimal Threshold: {optimal_threshold:.4f}, Accuracy: {accuracy:.4f}')
    return roc_auc, optimal_threshold, accuracy


def main():
    args = parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    train_set = LFW4Training(args.train_file, args.img_folder)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    eval_set = LFW4Eval(args.eval_file, args.img_folder)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size)

    # Instantiate model, loss, and optimizer
    model = SphereCNN(class_num = train_set.n_label).to(device)
    loss_fn = AngularPenaltySMLoss(in_features=512, out_features = train_set.n_label, m=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    # Use AverageMeter to track losses
    train_loss_meter = AverageMeter()
    eval_loss_meter = AverageMeter()

    for epoch in range(args.epoch):
        model.train()  # Set model to training mode
        train_loss_meter.reset()  # Reset AverageMeter at start of each epoch

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()  # Zero gradients
            outputs, _ = model(inputs, targets)  # Obtain outputs from the model
            loss = loss_fn(outputs, targets)  # Calculate loss
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update model parameters

            train_loss_meter.update(loss.item(), inputs.size(0))

        # Print average training loss
        print(f"Epoch: {epoch+1}/{args.epoch}, Training Loss: {train_loss_meter.avg}")


        # Evaluation logic
        model.eval()  # Set model to evaluation mode
        model.feature = True
        eval(eval_loader, model, device)
        model.feature = False
        eval_loss_meter.reset()  # Reset evaluation loss meter


        with torch.no_grad():
            for img_1, img_2, label in eval_loader:
                img_1, img_2, label = img_1.to(device), img_2.to(device), label.to(device)
                outputs_1 = model(img_1, None)
                outputs_2 = model(img_2, None)

                cos_sim = F.cosine_similarity(outputs_1, outputs_2)
                predicted_labels = (cos_sim > 0.5).long()
                correct_predictions = (predicted_labels == label).float().sum()
                accuracy = correct_predictions / label.size(0)

                eval_loss_meter.update(accuracy.item(), label.size(0))

        print(f"Epoch: {epoch+1}/{args.epoch}, Evaluation Accuracy: {eval_loss_meter.avg}")



if __name__ == "__main__":
    main()