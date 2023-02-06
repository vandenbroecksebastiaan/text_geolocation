import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from data import LocationDataset, collate_fn, find_characters_to_keep
from model import Model
from train import train

EPOCHS = 1000
LR = 0.005


def get_n_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return "Number of trainable parameters: " + str(params)


def main():
    dataset = LocationDataset()
    dataset.load_data()
    dataset.add_country_code()
    dataset.pre_process()
    dataset.to_tensor()

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True,
                            collate_fn=collate_fn)
    input_size = dataset.x_data[0].shape[-1]
    model = Model(input_size=input_size).cuda()
    print(get_n_parameters(model))
    model = train(model=model, train_loader=dataloader, EPOCHS=EPOCHS, LR=LR)


if __name__== "__main__":
    main()