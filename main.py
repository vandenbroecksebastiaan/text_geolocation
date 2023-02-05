import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from data import LocationDataset, collate_fn
from model import Model
from train import train



def main():
    dataset = LocationDataset()
    dataset.load_data()
    dataset.add_country_code()
    dataset.pre_process()
    dataset.to_tensor()

    dataloader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn)
    model = Model(input_size=37).cuda()
    model = train(model, dataloader, 10)


if __name__== "__main__":
    main()