import torch
import numpy as np


def train(model, train_loader, EPOCHS):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    model.train()

    predictions = torch.empty(size=(0, 2)).cuda()
    targets = torch.empty(size=(0, 2)).cuda()
    train_loss = torch.empty(size=(0, 1)).cuda()

    for epoch in range(EPOCHS):

        for idx, (x_train, y_train) in enumerate(train_loader):

            optimizer.zero_grad()
            output = model(x_train)
            loss = criterion(output[:, 0], y_train[:, 0]) \
                   + criterion(output[:, 1], y_train[:, 1])
            loss.backward()
            optimizer.step()

            print(epoch, idx, x_train.shape, loss.item(), end="\n")

            if idx % 10 == 0:
                predictions = torch.vstack([predictions, output])
                targets = torch.vstack([targets, y_train])
                train_loss = torch.vstack([train_loss, loss])

    np.savetxt("data/predictions.txt", predictions.detach().cpu().numpy())
    np.savetxt("data/targets.txt", targets.cpu().numpy())
    np.savetxt("data/train_loss.txt", train_loss.detach().cpu().numpy())