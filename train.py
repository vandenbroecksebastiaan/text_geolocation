from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import SGD, Adam
import numpy as np
from tqdm import tqdm
from geopy import distance
import wandb

from data import undo_min_max_scaling


def validation_step(model, eval_loader, criterion):
        model.eval()

        # Iterate over the evaluation dataloader
        eval_losses = []
        eval_predictions = np.empty(shape=(0, 6))
        eval_targets = np.empty(shape=(0, 6))

        for _, (input_ids, attention_mask, y_eval) in enumerate(eval_loader):
            eval_output = model(input_ids, attention_mask)
            eval_loss = criterion(eval_output, y_eval.argmax(dim=1)).item()
            eval_losses.append(eval_loss)

            eval_predictions = np.vstack([eval_predictions,
                                          eval_output.detach().cpu().numpy()])
            eval_targets = np.vstack([eval_targets,
                                      y_eval.detach().cpu().numpy()])

        # Undo the min max scaling on the coordinates
        eval_predictions_unscaled = undo_min_max_scaling(eval_predictions)
        eval_targets_unscaled = undo_min_max_scaling(eval_targets)

        # Calculate the distance between the prediction and target
        eval_distances = []
        for target, pred in zip(eval_predictions_unscaled, eval_targets_unscaled):
            if target[0] > 90: target[0] = 90
            if target[0] < 90: target[0] = -90
            if pred[0] > 90: target[0] = 90
            if pred[0] < 90: target[0] = -90
            eval_distance = distance.distance(target, pred).km
            eval_distances.append(eval_distance)

        mae_km = sum(eval_distances) / len(eval_distances)
        mean_eval_loss = sum(eval_losses) / len(eval_losses)

        return mae_km, mean_eval_loss


def train(model, train_loader, eval_loader, epochs, lr):

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss(weight=train_loader.dataset.dataset.weights.cuda())
    
    # wandb.init(project="text_geolocation")
    # wandb.watch(model, criterion, log="all", log_freq=len(train_loader))

    train_losses = []
    train_acc = []
    eval_losses = []
    eval_mae_km_list = []

    for epoch in tqdm(range(epochs), desc="Epoch"):

        epoch_train_acc = []
        epoch_train_losses = []

        # --- Train step
        model.train()
        for idx, (input_ids, attention_mask, y_train) in enumerate(train_loader):

            optimizer.zero_grad()
            train_output = model(input_ids, attention_mask)
            train_loss = criterion(train_output, y_train.argmax(dim=1))
            train_loss.backward()
            optimizer.step()
            
            # wandb.log({"loss": train_loss.item()}, step=epoch*idx)

            iteration_acc = (y_train.argmax(dim=1) == train_output.argmax(dim=1)).sum().item() / 32

            epoch_train_losses.append(train_loss.item())
            epoch_train_acc.append(iteration_acc)

            print(epoch, idx, "\t|", round(train_loss.item(), 4), "\t|",
                  round(iteration_acc, 4), "\t|", train_output.argmax(dim=1)[:10])
            
        train_losses.append(sum(epoch_train_losses) / len(epoch_train_losses))
        train_acc.append(sum(epoch_train_acc) / len(epoch_train_acc))

        # --- Validation step
        eval_mae_km, mean_eval_loss = validation_step(model, eval_loader, criterion)
        
        eval_mae_km_list.append(eval_mae_km)
        eval_losses.append(mean_eval_loss)
        
        print("Val mae km:", eval_mae_km, " | Mean val loss:", mean_eval_loss)

    """
    # After training, make a prediction on the train set
    train_predictions = np.empty(shape=(0, 2))
    train_targets = np.empty(shape=(0, 2))
    model.eval()
    for idx, (input_ids, attention_mask, y_train) in enumerate(train_loader):
         train_output = model(input_ids, attention_mask)
         train_predictions = np.vstack([train_predictions,
                                        train_output.detach().cpu().numpy()])
         train_targets = np.vstack([train_targets,
                                    y_train.detach().cpu().numpy()])
    """

    # After training, make a prediction on the validation set
    eval_predictions = np.empty(shape=(0, 1))
    eval_targets = np.empty(shape=(0, 1))
    model.eval()
    for idx, (input_ids, attention_mask, y_eval) in enumerate(eval_loader):
         eval_output = model(input_ids, attention_mask)
         eval_output = eval_output.argmax(dim=1).unsqueeze(0).t()
         eval_predictions = np.vstack([eval_predictions,
                                       eval_output.detach().cpu().numpy()])
         y_eval = y_eval.argmax(dim=1).unsqueeze(0).t()
         eval_targets = np.vstack([eval_targets,
                                   y_eval.detach().cpu().numpy()])

    # Save train data
    np.savetxt("data/train_losses.txt", np.array(train_losses, dtype=object), fmt="%f")
    # np.savetxt("data/train_predictions.txt", np.array(train_predictions, dtype=object), fmt="%f")
    # np.savetxt("data/train_targets.txt", np.array(train_targets, dtype=object), fmt="%f")

    # Save validation data
    np.savetxt("data/eval_losses.txt", np.array(eval_losses, dtype=object), fmt="%f")
    np.savetxt("data/eval_mae_km_list.txt", np.array(eval_mae_km_list, dtype=object), fmt="%f")
    np.savetxt("data/eval_predictions.txt", np.array(eval_predictions, dtype=object), fmt="%f")
    np.savetxt("data/eval_targets.txt", np.array(eval_targets, dtype=object), fmt="%f")