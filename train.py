import torch
from torch.nn import MSELoss
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

import numpy as np
from tqdm import tqdm


def train(model, train_loader, EPOCHS, LR):

    optimizer = SGD(model.parameters(), lr=LR)
    criterion = MSELoss()

    model.train()
    train_loss = []
    for epoch in range(EPOCHS):

        # --- Train step
        print(epoch)
        desc = "Training model"
        iteration_loss = []
        for idx, (x_train, y_train) in tqdm(enumerate(train_loader), desc=desc):

            optimizer.zero_grad()
            output = model(x_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

            iteration_loss.append(loss.item())

        mean_iteration_loss = sum(iteration_loss) / len(iteration_loss) 
        train_loss.append(mean_iteration_loss)
        print(" ", x_train.shape, round(mean_iteration_loss, 4), end="\r")

        # --- Validation step


    # Ask the trained model to make predictions and save them
    predictions = []
    targets = []
    model.eval()
    for idx, (x_train, y_train) in enumerate(train_loader):
        output = model(x_train)
        predictions.extend(output.tolist())
        targets.extend(y_train.tolist())

    np.savetxt("data/predictions.txt", np.array(predictions, dtype=object), fmt="%f")
    np.savetxt("data/targets.txt", np.array(targets, dtype=object), fmt="%f")
    np.savetxt("data/train_loss.txt", np.array(train_loss, dtype=object), fmt="%f")