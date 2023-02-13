import numpy as np
from torch.utils.data import DataLoader, random_split
import torch

from data import LocationDataset
from model import BaseRobertaModel, BaseRobertaLanguageDetectionModel
from train import train
from config import config


def get_n_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    print("-"*80)
    print("Number of trainable parameters:", str(n_params))
    print("-"*80)
    
# TODO: W&B implementation

def main():
    torch.set_printoptions(sci_mode=False)

    dataset = LocationDataset(max_obs=config["max_obs"], model_name=config["model_name"], n_classes=config["n_classes"])
    train_dataset, eval_dataset = random_split(dataset, lengths=[0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], drop_last=True, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config["batch_size"], drop_last=True, shuffle=True)
   
    model = BaseRobertaLanguageDetectionModel().cuda()
    get_n_parameters(model)
    train(model, train_loader, eval_loader, config["epochs"], config["lr"])
   

if __name__== "__main__":
    main()