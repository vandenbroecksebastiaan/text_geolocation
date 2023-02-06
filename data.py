import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from multiprocessing.pool import ThreadPool
import numpy as np
from tqdm import tqdm
import os
import json
import re
from unidecode import unidecode

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
    "Poland"        : "PL",
    "Canada"        : "CA",
    "United States" : "US"
}

code_to_country_map = {j:i for i,j in country_to_code_map.items()}



def collate_fn(tensor):
    x_batch = [i[0] for i in tensor]
    y_batch = torch.vstack([i[1] for i in tensor]).cuda()
    x_batch = pad_sequence(x_batch).cuda().squeeze()

    # Shorten long sequences
    if x_batch.shape[0] > 200:
        x_batch = x_batch[:200, :, :]

    return (x_batch, y_batch)


def obs_to_tensor(obs):
    """One hot encodes a string to a Tensor"""
    n_letters = len(characters_to_keep)
    tensor = torch.zeros(len(obs), 1, n_letters)
    for idx, char in enumerate(obs):
        tensor[idx][0][characters_to_keep.find(char)] = 1
    return tensor


def find_characters_to_keep(data, max_n_characters=100):
    """Finds the most frequent charachters in an iterable"""
    character_count_map = {}
    for obs in data:
        for char in obs:
            if char not in character_count_map.keys():
                character_count_map[char] = 0
            character_count_map[char] += 1

    character_count_sorted = sorted(character_count_map.items(),
                                 key=lambda x: x[1],
                                 reverse=True)

    global characters_to_keep
    characters_to_keep = "abcdefghijklmnopqrstuvwxyz"
    characters_to_keep += "12334567890"

    # Add the most frequent characters to characters_to_keep
    while (len(characters_to_keep) <= max_n_characters
          and len(character_count_sorted) > 0):
        new_char = character_count_sorted.pop(0)[0]
        if new_char not in characters_to_keep:
            characters_to_keep += new_char

    # Remove these characters regardless
    characters_to_remove = """ @.,#'_-":);(/&*`$%|[]+~><=^\{}"""
    for i in characters_to_remove:
        characters_to_keep = characters_to_keep.replace(i, "")

    print("Characters to keep: ", characters_to_keep)

    return characters_to_keep


class LocationDataset(Dataset):

    def __init__(self):
        self.x_data = []
        self.y_data = []

    def load_data(self):
        train_data_path = "data/training_data/training_data/"
        for file in tqdm(os.listdir(train_data_path), desc="Reading data"):
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
        # in_region = np.isin(self.country, list(country_to_code_map.values()))
        # self.x_data = self.x_data[in_region]
        # self.y_data = self.y_data[in_region]
        # self.country = self.country[in_region]
        
        # Replace line breaks and tabs
        self.x_data = np.char.replace(self.x_data, "\n", "")
        self.x_data = np.char.replace(self.x_data, "\t", "")

        # Unicode decode
        self.x_data = np.array([unidecode(i) for i in
                                tqdm(self.x_data, desc="Unidecode")])

        # All letters to lower case
        self.x_data = np.char.lower(self.x_data)

        # Replace links
        self.x_data = np.array([re.sub(r'http\S+', '', i) for i in self.x_data])

        # Remove/replace special characters
        self.characters_to_keep = find_characters_to_keep(self.x_data)
        with open("data/characters_to_keep.txt", "w") as file:
            file.write(self.characters_to_keep)

        for idx, obs in tqdm(enumerate(self.x_data), desc="Removing characters"):
            new_obs = ""
            for char in obs:
                if char in self.characters_to_keep:
                    new_obs += char
            self.x_data[idx] = new_obs

        # Order from short to long sentences for efficient batching
        obs_len = np.char.str_len(self.x_data)
        str_len_order = obs_len.argsort()
        self.x_data = self.x_data[str_len_order]
        self.y_data = self.y_data[str_len_order]
        self.country = self.country[str_len_order]

        # Remove empty and short observations
        obs_len = np.char.str_len(self.x_data)
        self.x_data = self.x_data[np.where(obs_len > 9)[0]]
        self.y_data = self.y_data[np.where(obs_len > 9)[0]]
        self.country = self.country[np.where(obs_len > 9)[0]]

        # Add weights based on country
        # class_weights_map = {} 
        # for country in np.unique(self.country):
        #     class_weights_map[country] = \
        #         len(self.country[self.country == country]) / len(self.country)

        # class_weights_map = {k:v/max(class_weights_map.values()) for k,v
        #                      in class_weights_map.items()}
        # self.weights = [class_weights_map[i] for i in self.country]

        # Reduce
        rand_idx = np.random.randint(low=0, high=len(self.x_data), size=500)
        self.x_data = self.x_data[rand_idx]
        self.y_data = self.y_data[rand_idx]
        self.country = self.country[rand_idx]
        # self.weights = self.weights[rand_idx]

    def to_tensor(self):
        """Transforms the strings x data into one hot encoded tensor on the
        character level"""
        with ThreadPool(10) as pool:
            x_data_encoded = pool.map(obs_to_tensor, self.x_data)

        self.x_data_raw = self.x_data
        self.x_data = x_data_encoded

    def add_country_code(self):
        coordinates = self.y_data.tolist()
        coordinates = [(i[1], i[0]) for i in coordinates]
        countries = reverse_geocoder.search(coordinates)
        self.country = np.array([i["cc"] for i in countries])

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        return (self.x_data[index], self.y_data[index]) 
