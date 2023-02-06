import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import numpy as np
from tqdm import tqdm


def train(model, train_loader, EPOCHS, LR):

    optimizer = Adam(model.parameters(), lr=LR)
    criterion = MSELoss()

    model.train()
    train_loss = torch.empty(size=(0, 1)).cuda()
    for epoch in range(EPOCHS):

        print(epoch)
        desc = "Training model"
        iteration_loss = []
        for idx, (x_train, y_train) in tqdm(enumerate(train_loader), desc=desc):

            optimizer.zero_grad()
            output = model(x_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

            iteration_loss.append(loss)

            print(" ", round(loss.item()), end="\r")

        train_loss = torch.vstack([train_loss, sum(iteration_loss) / len(iteration_loss)])

    # Ask the trained model to make predictions and save them
    predictions = torch.empty(size=(0, 2)).cuda()
    targets = torch.empty(size=(0, 2)).cuda()
    for idx, (x_train, y_train) in enumerate(train_loader):
        output = model(x_train)
        predictions = torch.vstack([predictions, output])
        targets = torch.vstack([targets, y_train])

    np.savetxt("data/predictions.txt", predictions.detach().cpu().numpy())
    np.savetxt("data/targets.txt", targets.cpu().numpy())
    np.savetxt("data/train_loss.txt", train_loss.detach().cpu().numpy())