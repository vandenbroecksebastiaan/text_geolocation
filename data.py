import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.multiprocessing

from multiprocessing.pool import Pool
import numpy as np
from tqdm import tqdm
import os
import json
import re

import reverse_geocoder

country_to_code_map = {
    "Belgium"       : "BE", 	
    "Greece"        : "EL", 	
    "ithuania"      : "LT", 	
    "Portugal"      : "PT",
    "Bulgaria"      : "BG", 
    "Spain"         : "ES", 	
    "Luxembourg"    : "LU", 	
    "Romania"       : "RO",
    "Czechia"       : "CZ", 	
    "France"        : "FR", 	
    "Hungary"       : "HU", 	
    "Slovenia"      : "SI",
    "Denmark"       : "DK", 	
    "Croatia"       : "HR", 	
    "Malta"         : "MT", 	
    "Slovakia"      : "SK",
    "Germany"       : "DE", 	
    "Italy"         : "IT", 	
    "Netherlands"   : "NL", 	
    "Finland"       : "FI",
    "Estonia"       : "EE", 	
    "Cyprus"        : "CY", 	
    "Austria"       : "AT", 
    "Sweden"        : "SE",
    "Ireland"       : "IE", 	
    "Latvia"        : "LV", 	
    "Poland"        : "PL"
    #"Canada"        : "CA",
    #"United States" : "US"
}

code_to_country_map = {j:i for i,j in country_to_code_map.items()}

characters_to_keep = "abcdefghijklmnopqrstuvwxyz"
characters_to_keep += "12334567890"

def collate_fn(tensor):             
    x_batch = [i[0] for i in tensor]
    y_batch = [i[1] for i in tensor]                       
    x_batch = pad_sequence(x_batch).cuda()
    y_batch = pad_sequence(y_batch).cuda()
    print(x_batch.shape)
    print(y_batch.shape)
    return (x_batch, y_batch)

def obs_to_tensor(obs):
    n_letters = len(characters_to_keep)
    tensor = torch.zeros(len(obs), 1, n_letters)
    for idx, char in enumerate(obs):
        tensor[idx][0][characters_to_keep.find(char)] = 1
    return tensor


class LocationDataset(Dataset):

    def __init__(self):
        self.x_data = []
        self.y_data = []

    def load_data(self):
        train_data_path = "data/training_data/training_data/"
        for file in tqdm(os.listdir(train_data_path)):
            with open(train_data_path + file, "r") as file:
                file_data = file.readlines()
                for obs in file_data:
                    obs_dict = json.loads(obs)
                    self.x_data.append(obs_dict["text"])
                    self.y_data.append(obs_dict["coordinates"])
    
        self.x_data = np.array(self.x_data)
        self.y_data = [(float(i[0]), float(i[1])) for i in self.y_data]
        self.y_data = torch.Tensor(self.y_data)

    def pre_process(self):
        # Remove countries that are not in the EU or NA
        in_region = np.isin(self.country, list(country_to_code_map.values()))
        self.x_data = self.x_data[in_region]
        self.y_data = self.y_data[in_region]
        self.country = self.country[in_region]
        
        # Replace line breaks and tabs
        self.x_data = np.char.replace(self.x_data, "\n", " ")
        self.x_data = np.char.replace(self.x_data, "\t", " ")

        # Replace links
        self.x_data = np.array([re.sub(r'http\S+', '', i) for i in self.x_data])

        # Remove/replace special characters
        for idx, obs in tqdm(enumerate(self.x_data)):
            new_obs = ""
            for char in obs:
                if char in characters_to_keep:
                    new_obs += char
            self.x_data[idx] = new_obs

        # Order from short to long sentences for efficient batching
        obs_len = np.char.str_len(self.x_data)
        str_len_order = obs_len.argsort()
        self.x_data = self.x_data[str_len_order]
        self.y_data = self.x_data[str_len_order]
        self.country = self.country[str_len_order]

        # Remove empty and short observations
        obs_len = np.char.str_len(self.x_data)
        self.x_data = self.x_data[np.where(obs_len > 9)[0]]
        self.y_data = self.y_data[np.where(obs_len > 9)[0]]
        self.country = self.country[np.where(obs_len > 9)[0]]

        np.savetxt("data/temp_x_preprocess.txt", self.x_data, fmt="%s")
    
    def to_tensor(self):
        """Transforms the strings x data into one hot encoded tensor on the
        character level"""

        # Multiprocessing using a pool
        torch.multiprocessing.set_sharing_strategy("file_system")
        print("opened pool")
        with Pool(10) as pool:
            x_data = pool.map(obs_to_tensor, self.x_data)
            print("here")
            pool.close()
        print("closed pool")

        self.x_data_encoded = x_data

    def add_country_code(self):
        coordinates = self.y_data.tolist()
        coordinates = [(i[1], i[0]) for i in coordinates]
        countries = reverse_geocoder.search(coordinates)
        self.country = np.array([i["cc"] for i in countries])

    def __len__(self):
        return len(self.x_data_encoded)

    def __getitem__(self, index):
        return (self.x_data_encoded[index], self.y_data[index]) 
