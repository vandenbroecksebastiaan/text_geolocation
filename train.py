import torch
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import SGD, Adam

import numpy as np
from tqdm import tqdm
from geopy import distance
import wandb

from data import undo_min_max_scaling
from config import config


def train(model, train_loader, eval_loader, epochs, lr):

    optimizer = Adam(model.parameters(), lr=lr)
    weight = train_loader.dataset.dataset.weights.cuda()
    criterion = CrossEntropyLoss(weight=weight)
    
    wandb.init(project="text_geolocation", config=config)
    wandb.watch(model, criterion, log="all", log_freq=len(train_loader) // 30)
    
    print("Len train loader:", len(train_loader))
    print("Len eval loader:", len(eval_loader))

    for epoch in tqdm(range(epochs), desc="Epoch"):

        for idx, (input_ids, attention_mask, y_train, target_coordinate) in enumerate(train_loader):
    
            # --- Train step
            model.train()
            optimizer.zero_grad()
            train_output = model(input_ids, attention_mask)
            train_loss = CrossEntropyLoss(train_output, y_train.argmax(dim=1))
            train_loss.backward()
            optimizer.step()

            iteration_acc = (y_train.argmax(dim=1) == train_output.argmax(dim=1)).sum().item() / config["batch_size"]
            
            if idx % 10 == 0: wandb.log({"train loss": train_loss.item(),
                                         "train acc": iteration_acc})

            # --- Validation step
            if idx % 100 == 0:
                eval_mae_km, mean_eval_loss = validation_step(model, eval_loader)
                
                print("Val mae km:", eval_mae_km, " | Mean val loss:", mean_eval_loss)
    
            print(epoch, idx, "\t|", round(train_loss.item(), 4),
                  "\t|", round(iteration_acc, 4))

    # After training, do a validation step to save last targets and predictions
    validation_step(model, eval_loader, write=True)


def validation_step(model, eval_loader, criterion, write=False):
        model.eval()
        cluster_to_coordinate_map = eval_loader.dataset.dataset.cluster_to_coordinate_map

        # Iterate over the evaluation dataloader
        eval_losses = []
        eval_predictions = np.empty(shape=(0, 2))
        eval_targets = np.empty(shape=(0, 2))

        for _, (input_ids, attention_mask, y_eval, target_coordinate) in enumerate(eval_loader):
            eval_output = model(input_ids, attention_mask).detach().cpu()
            eval_loss = criterion(eval_output, y_eval.argmax(dim=1)).item()

            eval_losses.append(eval_loss)
            eval_output_cluster = eval_output.argmax(dim=1).tolist()
            eval_output_coordinates = np.array(
                [cluster_to_coordinate_map[i] for i in eval_output_cluster]
            )
            
            target_coordinate = target_coordinate.cpu().numpy()
            eval_predictions = np.vstack([eval_predictions,
                                          eval_output_coordinates])
            eval_targets = np.vstack([eval_targets,
                                      target_coordinate])

        # Undo the min max scaling on the coordinates
        eval_predictions_unscaled = undo_min_max_scaling(eval_predictions)
        eval_targets_unscaled = undo_min_max_scaling(eval_targets)

        # Calculate the distance between the prediction and target
        eval_distances = []
        for target, pred in zip(eval_predictions_unscaled, eval_targets_unscaled):
            if target[0] > 90: target[0] = 90
            if target[0] < -90: target[0] = -90
            if pred[0] > 90: pred[0] = 90
            if pred[0] < -90: pred[0] = -90
            eval_distance = distance.distance(target, pred).km
            eval_distances.append(eval_distance)

        mae_km = sum(eval_distances) / len(eval_distances)
        mean_eval_loss = sum(eval_losses) / len(eval_losses)

        if write:
            np.savetxt("data/eval_predictions.txt",
                       np.array(eval_predictions_unscaled, dtype=object), fmt="%f")
            np.savetxt("data/eval_targets.txt",
                np.array(eval_targets_unscaled, dtype=object), fmt="%f"
            )
            
        wandb.log({"eval loss": mean_eval_loss, "mae km": mae_km})

        return mae_km, mean_eval_loss
