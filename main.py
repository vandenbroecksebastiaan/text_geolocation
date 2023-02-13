import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torch

from data import LocationDataset
from model import RobertaModel
from train import train


def get_n_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    print("-"*80)
    print("Number of trainable parameters:", str(n_params))
    print("-"*80)

EPOCHS = 5
LR = 0.00001

def main():
    torch.set_printoptions(sci_mode=False)

    dataset = LocationDataset(max_obs=20000)
    train_dataset, eval_dataset = random_split(dataset, lengths=[0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=32, drop_last=True, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=32, drop_last=True, shuffle=True)
   
    model = RobertaModel().cuda()
    get_n_parameters(model)
    train(model, train_loader, eval_loader, EPOCHS, LR)
   

if __name__== "__main__":
    main()