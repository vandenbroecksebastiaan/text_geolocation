import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from data import LocationDataset, collate_fn, find_characters_to_keep
from model import Model
from train import train

EPOCHS = 50
LR = 0.0001


def get_n_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("-"*30, "\n", "Number of trainable parameters: ", str(params),
          "\n", "-"*30)


def main():
    dataset = LocationDataset()
    dataset.load_data()
    dataset.add_country_code()
    dataset.reduce(max_obs=5000)
    dataset.min_max_scale_y_data()
    # Preprocessing
    dataset.remove_break_tab_link()
    dataset.unidecode()
    dataset.lower()
    dataset.remove_special_characters()
    dataset.sort()
    dataset.remove_short()
    dataset.write()
    dataset.to_tensor()

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True,
                            collate_fn=collate_fn)
    input_size = dataset.x_data[0].shape[-1]
    model = Model(input_size=input_size).cuda()
    get_n_parameters(model)
    model = train(model=model, train_loader=dataloader, EPOCHS=EPOCHS, LR=LR)


if __name__== "__main__":
    main()