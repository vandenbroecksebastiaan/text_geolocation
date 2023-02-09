import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torchtext
import torch

from data import LocationDataset, collate_fn, find_characters_to_keep
from model import Model, RobertaModel
from train import train

EPOCHS = 500
LR = 0.001


def get_n_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("-"*30, "\n", "Number of trainable parameters: ", str(params),
          "\n", "-"*30)

# TODO: tokenize tags since they may be informative for a region?

def main():
    torch.set_printoptions(sci_mode=False)

    dataset = LocationDataset(max_obs=100)
    dataloader = DataLoader(dataset, batch_size=4, drop_last=True)
    
    idx, (input_id, attention_mask, continent) = next(enumerate(dataloader))
   
    model = RobertaModel()
    output = model(input_id, attention_mask)
    print(output)
   
    
    # vocab = torchtext.vocab.build_vocab_from_iterator(
    #     [i.split() for i in dataset.x_data], max_tokens=5000
    # )
    
    # dataset.find_most_popular_words()
    
    # exit(0)
    
    # dataset.to_tensor()
    
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=False,
    #                         collate_fn=collate_fn)
    # input_size = dataset.x_data[0].shape[-1]
    # model = Model(input_size=input_size).cuda()
    # get_n_parameters(model)
    # model = train(model=model, train_loader=dataloader, EPOCHS=EPOCHS, LR=LR)


if __name__== "__main__":
    main()