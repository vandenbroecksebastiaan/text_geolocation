import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from data import LocationDataset, collate_fn



def main():
    dataset = LocationDataset()
    dataset.load_data()
    dataset.add_country_code()
    dataset.pre_process()
    dataset.to_tensor()

    dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)

    sample = next(enumerate(dataloader))
    print(sample.shape)

if __name__== "__main__":
    main()